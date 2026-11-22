//! One Sculpt job at a time: the panel's one lightweight analysis, then its previews. The panel
//! submits the newest slider values each frame; while a slider moves the running preview is left
//! to finish, so the canvas keeps updating during the drag, and once the values settle
//! ([`SETTLE_MS`]) a preview still running for older ones is cancelled so the last value is not
//! held up — the same rule as the filter dialogs' `filter_preview_worker`.
//!
//! A preview is a pure function of its [`Job`]: the engine's own `field_for` (the one Apply
//! builds) over a proxy of the target layer. No session, no document lock: a stale frame can
//! never be committed. The opening analysis runs here too ([`analyze`]), from an `Arc` document
//! and the engine's own `analyze_anchors`, so the panel paints at once instead of waiting on
//! the models.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, TryRecvError};

use egui::ColorImage;
use photocraft_algo::liquify::{LiquifyField, ProxyImage};
use photocraft_engine::jobs::JobCtx;
use photocraft_geom::Rect;
use serde_json::Value;

/// The tallest strip a render takes before it checks for a cancel: below the proxy renderer's
/// rayon threshold, so every strip stays on this thread and a superseded frame stops quickly.
const STRIP_ROWS: usize = 64;

/// What a superseded job answers; a cancel is not a failure to report.
pub(crate) const SUPERSEDED: &str = "the preview was superseded";

/// How long the panel's values must stay unchanged before a stale preview is cancelled. Shorter
/// than a pause in a drag is long; cancelling on every slider step meant no preview finished
/// until the slider stopped.
pub(crate) const SETTLE_MS: f64 = 150.0;

/// The pixels one request renders, premultiplied: the whole document's proxy and, for the
/// face panel, the face's own near-1:1 crop drawn over it.
pub(crate) struct Frame {
    pub image: ColorImage,
    pub crop: Option<CropFrame>,
}

/// The face's crop: the proxy pixels and the document rectangle they quad covers.
pub(crate) struct CropFrame {
    pub image: ColorImage,
    pub bounds: Rect,
}

/// What the crop warps: a proxy over the face's own rectangle, at one pixel per document pixel
/// where the face allows, and the selection's coverage at its pixels.
pub(crate) struct CropJob {
    pub proxy: Arc<ProxyImage>,
    pub selection: Option<Arc<Vec<f32>>>,
}

/// The panel's opening analysis: the anchors Apply will carry, the measured body fit (Body
/// only), plus Face Sculpt's quality reading (where the face box came from and the mesh's
/// geometry).
pub(crate) struct Analysis {
    pub anchors: Value,
    pub fit: Option<Value>,
    pub contour: Option<Value>,
    pub quality: Option<Value>,
    /// The layer whose own pixels were sampled, or `None` when the analysis read the flattened
    /// document — Apply cannot warp the latter.
    pub layer: Option<u64>,
}

/// What a finished job produced.
pub(crate) enum Output {
    Frame(Frame),
    Analysis(Analysis),
}

/// One preview request: the params Apply would send (anchors included), the document's extent,
/// the proxy to warp and the selection's coverage at its pixels.
pub(crate) struct Job {
    pub params: Value,
    pub canvas: Rect,
    pub face: bool,
    pub proxy: Arc<ProxyImage>,
    /// Coverage at each proxy pixel, row-major, when a selection limits the effect.
    pub selection: Option<Arc<Vec<f32>>>,
    /// Face Sculpt's near-1:1 crop, drawn over the whole-document proxy.
    pub crop: Option<CropJob>,
}

/// The proxy pixels through `job`'s field and selection, as the texture shows them. The field
/// is built once and sampled at each proxy's own document coordinates, so the crop and the
/// whole-document proxy show the same warp at different resolutions.
pub(crate) fn render(job: &Job, ctx: &JobCtx) -> Result<Output, String> {
    let field = if job.face {
        photocraft_engine::face_sculpt_cmds::field_for(&job.params, job.canvas).map_err(|e| e.to_string())?.0
    } else {
        photocraft_engine::sculpt_cmds::field_for(&job.params, job.canvas).map_err(|e| e.to_string())?.0
    };
    let image = pixels(render_proxy(&job.proxy, job.selection.as_deref(), &field, ctx)?, job.proxy.w, job.proxy.h);
    let crop = match &job.crop {
        Some(c) => {
            ctx.check().map_err(|_| SUPERSEDED.to_string())?;
            Some(CropFrame { image: pixels(render_proxy(&c.proxy, c.selection.as_deref(), &field, ctx)?, c.proxy.w, c.proxy.h), bounds: c.proxy.bounds })
        }
        None => None,
    };
    Ok(Output::Frame(Frame { image, crop }))
}

/// One proxy through `field`, in row strips so a superseded job stops between them.
fn render_proxy(proxy: &ProxyImage, selection: Option<&Vec<f32>>, field: &LiquifyField, ctx: &JobCtx) -> Result<Vec<[u8; 4]>, String> {
    let (w, h) = (proxy.w, proxy.h);
    let mut out = vec![[0u8; 4]; w * h];
    let mut y = 0;
    while y < h {
        let y1 = (y + STRIP_ROWS).min(h);
        proxy.render(field, [0, y, w, y1], &mut out);
        y = y1;
        ctx.check().map_err(|_| SUPERSEDED.to_string())?;
    }
    if let Some(cov) = selection {
        for ((px, old), &c) in out.iter_mut().zip(&proxy.px).zip(cov.iter()) {
            *px = mix(*px, *old, c);
        }
    }
    Ok(out)
}

fn pixels(out: Vec<[u8; 4]>, w: usize, h: usize) -> ColorImage {
    let pixels = out.iter().map(|p| egui::Color32::from_rgba_premultiplied(p[0], p[1], p[2], p[3])).collect();
    ColorImage::new([w, h], pixels)
}

/// One opening analysis: the document to read, the layer Apply will warp, which landmarks and
/// the user's drawn face box, if any.
pub(crate) struct AnalysisJob {
    pub doc: Arc<photocraft_doc::Document>,
    pub layer: photocraft_doc::LayerId,
    pub kind: photocraft_engine::portrait_cmds::SculptKind,
    pub roi: Option<Rect>,
}

/// `job`'s anchors, through the engine's own `analyze_anchors` and this job's cancellation.
pub(crate) fn analyze(job: &AnalysisJob, ctx: &JobCtx) -> Result<Output, String> {
    let found = photocraft_engine::portrait_cmds::analyze_anchors(&job.doc, Some(job.layer), false, job.kind, job.roi, ctx).map_err(|e| e.to_string())?;
    let contour = found.contour.map(serde_json::to_value).transpose().map_err(|e| format!("the contour would not serialise: {e}"))?;
    Ok(Output::Analysis(Analysis { anchors: found.anchors, fit: found.fit, contour, quality: found.quality, layer: Some(job.layer.0) }))
}

/// `new` over `old` (premultiplied) by coverage `c`, as the engine mixes the committed warp
/// (`distort_cmds::mix_by_selection`); at proxy resolution the seam is approximate, the commit is not.
fn mix(new: [u8; 4], old: [f32; 4], c: f32) -> [u8; 4] {
    let c = c.clamp(0.0, 1.0);
    std::array::from_fn(|k| (f32::from(new[k]) * c + old[k].clamp(0.0, 1.0) * 255.0 * (1.0 - c) + 0.5).clamp(0.0, 255.0) as u8)
}

/// The job being computed: its request key, its result channel and its cancellation.
struct Running {
    key: u64,
    rx: Receiver<Result<Output, String>>,
    job: JobCtx,
}

#[derive(Default)]
pub(crate) struct Worker {
    running: Option<Running>,
    /// The newest request and when it was first seen (ms).
    latest: Option<(u64, f64)>,
}

impl Worker {
    pub fn busy(&self) -> bool {
        self.running.is_some()
    }

    /// Never wait for a previous computation and never spawn a second concurrent job.
    /// `compute` gets the job's cancellation context ([`Worker::supersede`]).
    pub fn start(&mut self, key: u64, ctx: egui::Context, compute: impl FnOnce(&JobCtx) -> Result<Output, String> + Send + 'static) -> Result<bool, String> {
        if self.busy() {
            return Ok(false);
        }
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let job = JobCtx::new();
        let work = job.clone();
        std::thread::Builder::new()
            .name("sculpt-preview".into())
            .spawn(move || {
                let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compute(&work))) {
                    Ok(result) => result,
                    Err(_) => Err("The Sculpt task failed".to_string()),
                };
                let _ = tx.send(result);
                ctx.request_repaint();
            })
            .map_err(|e| format!("Could not start the Sculpt task: {e}"))?;
        self.running = Some(Running { key, rx, job });
        Ok(true)
    }

    /// The panel wants `latest` at time `now_ms`. Once that request has been unchanged for
    /// [`SETTLE_MS`], a running preview for anything else is cancelled, so it stops at its next
    /// check and the latest can start (the caller discards its result). While the request keeps
    /// changing (a slider drag) the running preview is left to finish and be shown.
    pub fn supersede(&mut self, latest: u64, now_ms: f64) {
        let since = match &self.latest {
            Some((key, t)) if *key == latest => *t,
            _ => {
                self.latest = Some((latest, now_ms));
                now_ms
            }
        };
        if now_ms - since < SETTLE_MS {
            return;
        }
        if let Some(r) = &self.running
            && r.key != latest
        {
            r.job.cancel();
        }
    }

    pub fn poll(&mut self) -> Option<(u64, Result<Output, String>)> {
        let result = match self.running.as_ref()?.rx.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Disconnected) => Err("Sculpt worker disconnected".to_string()),
        };
        let key = self.running.take()?.key;
        Some((key, result))
    }

    /// Stop the running preview (the panel closed, was applied, or the document changed).
    pub fn cancel(&mut self) {
        if let Some(r) = &self.running {
            r.job.cancel();
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> Output {
        Output::Frame(Frame { image: ColorImage::new([1, 1], vec![egui::Color32::BLACK]), crop: None })
    }

    fn finish(worker: &mut Worker) -> (u64, Result<Output, String>) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if let Some(result) = worker.poll() {
                return result;
            }
            assert!(std::time::Instant::now() < deadline, "preview worker did not finish");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// A slider drag submits every frame, but the worker runs one preview at a time: the
    /// intermediate requests are not queued, and the UI keeps drawing.
    #[test]
    fn blocked_preview_does_not_queue_slider_requests() {
        let mut worker = Worker::default();
        let ctx = egui::Context::default();
        let (release, wait) = std::sync::mpsc::channel();
        assert!(
            worker
                .start(1, ctx.clone(), move |_| {
                    wait.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
                    Ok(frame())
                })
                .unwrap()
        );
        for key in [2, 3, 4, 5] {
            ctx.run_ui(Default::default(), |ui| {
                ui.label("The UI still draws");
            })
            .textures_delta
            .clear();
            assert!(worker.poll().is_none());
            assert!(!worker.start(key, ctx.clone(), |_| panic!("queued an intermediate request")).unwrap());
        }
        release.send(()).unwrap();
        assert_eq!(finish(&mut worker).0, 1);
        assert!(worker.start(5, ctx, |_| Ok(frame())).unwrap());
        assert_eq!(finish(&mut worker).0, 5);
    }

    /// A panic inside a preview is an error and the worker stays usable for the next slider value.
    #[test]
    fn escaped_panic_is_an_error_and_worker_can_be_reused() {
        let mut worker = Worker::default();
        worker.start(1, Default::default(), |_| panic!("synthetic preview failure")).unwrap();
        assert!(finish(&mut worker).1.is_err());
        assert!(!worker.busy());
        worker.start(2, Default::default(), |_| Ok(frame())).unwrap();
        assert!(finish(&mut worker).1.is_ok());
    }

    /// Cancelling on every slider step meant no preview finished until the slider stopped; the
    /// stale preview must be cancelled only once the values settle.
    #[test]
    fn a_stale_preview_is_cancelled_only_once_the_values_settle() {
        let mut worker = Worker::default();
        let (send_job, job) = std::sync::mpsc::channel();
        let (release, wait) = std::sync::mpsc::channel::<()>();
        worker
            .start(1, Default::default(), move |ctx| {
                send_job.send(ctx.clone()).unwrap();
                let _ = wait.recv_timeout(std::time::Duration::from_secs(5));
                Ok(frame())
            })
            .unwrap();
        let job = job.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        for (i, key) in [2, 3, 4, 5].into_iter().enumerate() {
            worker.supersede(key, 1000.0 + i as f64 * 16.0);
            assert!(!job.cancelled(), "cancelled mid-drag at {key}");
        }
        worker.supersede(5, 1048.0 + SETTLE_MS - 1.0);
        assert!(!job.cancelled());
        worker.supersede(5, 1048.0 + SETTLE_MS);
        assert!(job.cancelled());
        release.send(()).unwrap();
        assert_eq!(finish(&mut worker).0, 1);
        // A running preview for the settled request itself is never cancelled.
        let (send_job, job) = std::sync::mpsc::channel();
        worker
            .start(5, Default::default(), move |ctx| {
                send_job.send(ctx.clone()).unwrap();
                Ok(frame())
            })
            .unwrap();
        let job = job.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        worker.supersede(5, 99_999.0);
        assert!(!job.cancelled());
        let _ = finish(&mut worker);
    }
}
