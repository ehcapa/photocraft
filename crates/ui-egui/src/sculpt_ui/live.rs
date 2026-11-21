//! The Sculpt panels' one portrait analysis and their live canvas preview.
//!
//! Opening a panel runs `portrait.analyze` once — the pose or face-mesh anchors, never the full
//! Smart Portrait pipeline — and Apply hands those anchors to the engine command, so an Apply
//! builds the field and warps, nothing more. While a slider moves, a background job renders the
//! same field (the engine's own `field_for`, what Apply builds) over a proxy of the target layer;
//! the canvas hides that layer and draws the warped proxy in its place. Apply stays exact: the
//! preview is approximate by design (proxy resolution, no mask, effects or blend), and wherever
//! it would show a warp the composite would not, it is suppressed with an inline note instead.
//!
//! The browser build has no threads, so it has no live preview; Apply works there as everywhere.

use std::sync::Arc;

use craft::behavior::kurbo::Point;
use craft::imaging::ui::transform_preview::{PreviewPixels, PreviewTextures};
use craft::imaging::ui::{PreviewMap, TransformPreview};
use craft::ui::contract::Show as _;
use photocraft_doc::{BlendMode, DocId, Document, LayerContent, LayerId, LayerMask};
use photocraft_engine::portrait_cmds::SculptKind;
use photocraft_geom::Rect;
use serde_json::{Value, json};

#[cfg(not(target_arch = "wasm32"))]
use photocraft_algo::liquify::ProxyImage;
#[cfg(not(target_arch = "wasm32"))]
use photocraft_raster::Surface;

use crate::PhotocraftApp;

use super::Panel;

#[cfg(not(target_arch = "wasm32"))]
mod worker;

/// The proxy's longest side. It is sharp at fit zoom and one render costs a few tens of
/// milliseconds; the proxy is rebuilt only when the document's pixels change.
const PROXY_SIDE: usize = 1600;

/// The preview's share of the canvas display keys: `distort_ui` uses 1 << 42, Free Transform
/// 1 << 40.
const DISPLAY_KEY: u64 = 1 << 43;

/// The analysis result, the preview runtime and the target the panel applies to.
pub struct Live {
    /// The command the panel applies with.
    command: &'static str,
    /// Names the preview's texture and folds into the canvas display key.
    serial: u64,
    doc: Option<DocId>,
    layer: Option<LayerId>,
    size: [u32; 2],
    /// The document revision the anchors and the proxy were read at.
    revision: u64,
    /// `portrait.analyze`'s anchors, `Null` when the analysis failed.
    anchors: Value,
    /// Body Sculpt's measured fit: the visible silhouette the field widths come from. `None`
    /// for Face, or when the mask has no measurable silhouette.
    fit: Option<Value>,
    contour: Option<Value>,
    /// Face Sculpt's quality reading: the face box's source and the mesh's geometry. `None`
    /// for Body, or when the analysis failed.
    quality: Option<Value>,
    /// The user's drawn face box, if any: the next analysis meshes inside it.
    roi: Option<Rect>,
    /// Whether the analysis sampled the bound layer's own pixels: the layer Apply can warp.
    sampled: bool,
    /// The bound layer is a raster surface or a smart object — the kinds Apply can warp.
    warpable: bool,
    /// The opening analysis is still running: the panel paints, Apply waits.
    busy: bool,
    /// Why the analysis failed, or `None`.
    notice: Option<String>,
    /// Why a preview could not be built although the panel can apply, or `None`.
    preview_error: Option<String>,
    /// Why the live preview is off although the panel can apply, or `None`.
    suppress: Option<&'static str>,
    /// The panel's params the newest preview request was made for (or the values last applied):
    /// unchanged values never re-preview, so an Apply's warp is not drawn a second time.
    last_requested: Option<Value>,
    /// The newest request's identity, advanced on every panel change; a running job for an older
    /// identity is superseded once the values settle.
    generation: u64,
    /// A request is wanted but could not start because the worker was busy.
    want: bool,
    /// The identity and revision of the running request.
    pending: Option<(u64, u64)>,
    preview: Option<Preview>,
    #[cfg(not(target_arch = "wasm32"))]
    proxy: Option<CachedProxy>,
    /// The opening analysis in flight: its document, layer and the revision it read.
    #[cfg(not(target_arch = "wasm32"))]
    analysis_pending: Option<(DocId, LayerId, u64)>,
    #[cfg(not(target_arch = "wasm32"))]
    worker: worker::Worker,
}

/// A published frame: the texture the canvas draws and the document it hides the layer in.
struct Preview {
    textures: PreviewTextures,
    /// The document with the target layer hidden.
    without: Arc<Document>,
    /// The layer's opacity times its fill.
    opacity: f32,
    size: [u32; 2],
}

/// The proxy of the target layer at one document revision, with the selection's coverage at its
/// pixels. Rebuilt only when the revision moves.
#[cfg(not(target_arch = "wasm32"))]
struct CachedProxy {
    revision: u64,
    proxy: Arc<ProxyImage>,
    selection: Option<Arc<Vec<f32>>>,
}

/// What a job warps: the proxy and the selection's per-pixel coverage, if one limits the effect.
#[cfg(not(target_arch = "wasm32"))]
type PreviewSource = (Arc<ProxyImage>, Option<Arc<Vec<f32>>>);

impl Live {
    /// The panel's runtime: one lightweight analysis at the active layer, started in the
    /// background so the panel paints at once and Apply waits for the anchors.
    pub(crate) fn open(app: &mut PhotocraftApp, command: &'static str) -> Live {
        let serial = app.ui.alloc_id();
        let mut live = Live {
            command,
            serial,
            doc: None,
            layer: None,
            size: [0, 0],
            revision: 0,
            anchors: Value::Null,
            fit: None,
            contour: None,
            quality: None,
            roi: None,
            sampled: false,
            warpable: false,
            busy: false,
            notice: None,
            preview_error: None,
            suppress: None,
            // A freshly opened panel's params are `{}`: an untouched panel previews nothing.
            last_requested: Some(json!({})),
            generation: 0,
            want: false,
            pending: None,
            preview: None,
            #[cfg(not(target_arch = "wasm32"))]
            proxy: None,
            #[cfg(not(target_arch = "wasm32"))]
            analysis_pending: None,
            #[cfg(not(target_arch = "wasm32"))]
            worker: worker::Worker::default(),
        };
        let Some(st) = app.session.active() else {
            live.notice = Some(tl!("Open a document to sculpt").into());
            return live;
        };
        live.doc = Some(st.doc.id);
        live.layer = st.active_layer;
        live.size = [st.doc.size.width, st.doc.size.height];
        live.revision = st.revision;
        // The Apply path warps only a pixel layer's or a smart object's own surface. A layer of
        // another kind can still be sampled (its cached rendering), but not sculpted — no
        // analysis is worth running to disable Apply.
        live.warpable = live.layer.is_some_and(|id| matches!(st.doc.layer(id).map(|l| &l.content), Some(LayerContent::Raster(_) | LayerContent::Smart(_))));
        if !live.warpable {
            live.notice = Some(tl!("Select a pixel layer or smart object to sculpt").into());
            return live;
        }
        #[cfg(not(target_arch = "wasm32"))]
        live.start_analysis(app);
        #[cfg(target_arch = "wasm32")]
        live.analyze_sync(app);
        live
    }

    /// The landmark set this panel asks for.
    fn kind(&self) -> SculptKind {
        if self.command == super::FACE { SculptKind::Face } else { SculptKind::Body }
    }

    /// Whether the opening analysis is still in flight.
    pub fn busy(&self) -> bool {
        self.busy
    }

    /// Starts the opening analysis on the worker; Apply stays disabled until [`Live::poll`]
    /// takes its answer. The document is `Arc`-shared, so the worker reads the pixels the
    /// panel was opened on without locking the editor.
    #[cfg(not(target_arch = "wasm32"))]
    fn start_analysis(&mut self, app: &mut PhotocraftApp) {
        let (Some(doc_id), Some(layer)) = (self.doc, self.layer) else { return };
        let Some(st) = app.session.active() else { return };
        if st.doc.id != doc_id {
            return;
        }
        let job = worker::AnalysisJob { doc: Arc::clone(&st.doc), layer, kind: self.kind(), roi: self.roi };
        self.busy = true;
        self.analysis_pending = Some((doc_id, layer, self.revision));
        match self.worker.start(self.serial, app.ctx().clone(), move |ctx| worker::analyze(&job, ctx)) {
            Ok(true) => {}
            // The worker was still finishing another job (a rebind); tick retries.
            Ok(false) => {}
            Err(e) => self.fail(e),
        }
    }

    /// The browser's path: no threads, so the analysis runs inline like it used to everywhere.
    #[cfg(target_arch = "wasm32")]
    fn analyze_sync(&mut self, app: &mut PhotocraftApp) {
        let mut params = match self.layer {
            Some(layer) => json!({"kind": self.kind().as_str(), "layer": layer.0}),
            None => json!({"kind": self.kind().as_str()}),
        };
        if let Some(r) = self.roi {
            params["roi"] = json!([r.x0, r.y0, r.width(), r.height()]);
        }
        self.busy = true;
        match app.session.execute("portrait.analyze", params) {
            Ok(reply) => {
                let anchors = reply.get("anchors").cloned().unwrap_or(Value::Null);
                let fit = reply.get("fit").cloned();
                let contour = reply.get("contour").cloned();
                let quality = reply.get("quality").cloned();
                let layer = reply.get("layer").and_then(Value::as_u64);
                self.finish(app, anchors, fit, contour, quality, layer);
            }
            Err(e) => self.fail(e.to_string()),
        }
    }

    /// The analysis's answer into the panel's state: the anchors Apply will carry, the body
    /// fit the field widths come from, the quality the panel warns from, and the inline notice
    /// when it found nothing or sampled a layer Apply cannot warp.
    fn finish(&mut self, app: &PhotocraftApp, anchors: Value, fit: Option<Value>, contour: Option<Value>, quality: Option<Value>, layer: Option<u64>) {
        self.busy = false;
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.analysis_pending = None;
        }
        self.anchors = anchors;
        self.fit = fit;
        self.contour = contour;
        self.quality = quality;
        // `layer` is null when the analysis read the flattened document: the bound layer then
        // has no pixels of its own to warp.
        self.sampled = self.warpable && layer == self.layer.map(|l| l.0);
        if !self.anchors.is_object() {
            self.notice = Some(tl!("The analysis found no landmarks").into());
        } else if !self.sampled {
            self.notice = Some(tl!("Select a pixel layer or smart object to sculpt").into());
        } else {
            self.notice = None;
        }
        if let Some(st) = app.session.active() {
            self.revision = st.revision;
        }
    }

    /// The analysis's refusal inline: the sculpt panel's own wording, or the engine's error.
    fn fail(&mut self, error: impl Into<String>) {
        self.busy = false;
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.analysis_pending = None;
        }
        let error = error.into();
        self.notice = Some(if error.contains("no person") {
            tl!("No person found — Body Sculpt needs the body-pose landmarks").into()
        } else if error.contains("no face") {
            tl!("No face found — Face Sculpt needs the face-mesh landmarks").into()
        } else {
            error
        });
    }

    /// Whether Apply has the anchors and the layer it needs.
    pub fn ready(&self) -> bool {
        self.anchors.is_object() && self.layer.is_some() && self.sampled
    }

    /// The line the panel shows under its tabs: the analysis's refusal, else why the live
    /// preview is off. `None` when both are fine.
    pub fn notice_text(&self) -> Option<String> {
        if let Some(notice) = &self.notice {
            return Some(notice.clone());
        }
        if let Some(error) = &self.preview_error {
            return Some(error.clone());
        }
        self.suppress.map(|why| crate::i18n::t(why).to_owned())
    }

    /// The question the analysis raises before the warp is trusted: for Face, the
    /// whole-document fallback (no face box) or a turned head; for Body, feet the pose did not
    /// see — Height and Leg Length need the ankles, and without them the sliders cannot move
    /// the legs at all. `None` when the analysis is fine, or before one has landed.
    pub fn warning_text(&self) -> Option<String> {
        if let Some(q) = &self.quality {
            if q.get("source").and_then(Value::as_str) == Some("wholeDocument") {
                return Some(tl!("No face box was found, so the landmarks came from the whole photo; select the face and re-analyse for a closer fit").into());
            }
            let turn = q.get("turnDeg").and_then(Value::as_f64).unwrap_or(0.0);
            return (turn.abs() > 20.0).then(|| tl!("The face is turned; Face Sculpt is least accurate on a profile").to_string());
        }
        // The same gate the field uses (the engine's pose visibility floor): below it the pose
        // is guessing, and the length rows are skipped rather than stretching the wrong part.
        const SEEN: f64 = 0.5;
        let pose = self.anchors.get("pose")?.as_array()?;
        let seen = |i: usize| pose.get(i).and_then(|p| p.get(2)).and_then(Value::as_f64).unwrap_or(0.0) >= SEEN;
        (!seen(27) || !seen(28)).then(|| tl!("The feet are not in view, so Height and Leg Length cannot lengthen the legs").to_string())
    }

    /// The face box the current anchors were meshed inside, if the user drew one.
    pub(crate) fn roi(&self) -> Option<Rect> {
        self.roi
    }

    /// Runs the analysis again, inside `roi` when one is given: the panel's response to a
    /// warning, or to a new selection. The old anchors and frame go at once; Apply waits for
    /// the new answer.
    pub(crate) fn reanalyze(&mut self, app: &mut PhotocraftApp, roi: Option<Rect>) {
        self.suspend();
        self.roi = roi;
        self.anchors = Value::Null;
        self.fit = None;
        self.contour = None;
        self.quality = None;
        self.sampled = false;
        self.notice = None;
        self.preview_error = None;
        self.last_requested = None;
        self.generation = self.generation.wrapping_add(1);
        self.want = false;
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.worker.busy() {
                // Let the cancelled job drain first: tick starts the analysis as soon as the
                // worker is free, so a cancelled preview cannot be mistaken for a cancelled
                // analysis.
                self.busy = true;
                self.analysis_pending = None;
            } else {
                self.start_analysis(app);
            }
        }
        #[cfg(target_arch = "wasm32")]
        self.analyze_sync(app);
    }

    /// The document this panel belongs to, once one was found.
    pub(crate) fn document(&self) -> Option<DocId> {
        self.doc
    }

    /// The document's extent, the field's canvas.
    fn canvas(&self) -> Rect {
        Rect::new(0, 0, self.size[0] as i32, self.size[1] as i32)
    }

    /// Drops the shown frame and the running request: the canvas goes back to the document.
    fn suspend(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        self.worker.cancel();
        self.pending = None;
        self.preview = None;
    }

    /// One frame of the panel's runtime: follows the document and the target layer, drops a
    /// preview the document no longer matches, and starts or supersedes the preview job.
    pub(crate) fn tick(&mut self, app: &mut PhotocraftApp, panel: &Panel, ctx: &egui::Context) {
        let Some(st) = app.session.active() else {
            self.suspend();
            return;
        };
        if self.doc.is_none() {
            // Opened before a document existed: this one is the target now.
            let command = self.command;
            *self = Live::open(app, command);
            self.last_requested = None;
            return;
        }
        if Some(st.doc.id) != self.doc {
            // The shell closes the panel; the canvas must not keep a foreign preview meanwhile.
            self.suspend();
            return;
        }
        let size = [st.doc.size.width, st.doc.size.height];
        let revision = st.revision;
        let active_layer = st.active_layer;
        // A resized document moves every anchor, and another active layer has its own pose or
        // face: analyse again, keeping the panel's slider values and locks.
        if size != self.size || active_layer != self.layer {
            let command = self.command;
            *self = Live::open(app, command);
            self.last_requested = None;
            return;
        }
        let Some(layer_id) = self.layer else {
            self.suspend();
            return;
        };
        // Ends the session borrow before `poll` needs the app mutably.
        let layer_present = st.doc.layer(layer_id).is_some();
        if !layer_present {
            self.notice = Some(tl!("The sculpt target layer is gone").into());
            self.anchors = Value::Null;
            self.fit = None;
            self.sampled = false;
            self.suspend();
            return;
        }
        // Poll before the readiness gate: the opening analysis's answer arrives while the panel
        // has no anchors yet. A worker that frees up without a running analysis (a failed start)
        // gets the request again.
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.poll(app, ctx);
            if self.busy && !self.worker.busy() {
                self.start_analysis(app);
            }
        }
        if self.busy {
            // Waiting on the analysis: nothing to preview, and suspending would cancel it.
            return;
        }
        if !self.ready() {
            self.suspend();
            return;
        }
        // Any other edit (paint, undo, a committed command) moved the pixels the anchors and the
        // proxy were read from: the shown frame is stale, and the proxy must be rebuilt from the
        // edited pixels when the next preview starts.
        if revision != self.revision {
            self.revision = revision;
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.proxy = None;
            }
            self.preview_error = None;
            self.suspend();
        }
        let doc = {
            let Some(st) = app.session.active() else {
                self.suspend();
                return;
            };
            Arc::clone(&st.doc)
        };
        self.suppress = suppression(app, &doc, layer_id);
        if self.suppress.is_some() {
            self.suspend();
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let params = panel.params();
            if self.last_requested.as_ref() != Some(&params) {
                self.last_requested = Some(params);
                self.generation = self.generation.wrapping_add(1);
                self.want = true;
            }
            if self.want && self.start_job(app, ctx) {
                self.want = false;
            }
            let now_ms = ctx.input(|i| i.time) * 1000.0;
            self.worker.supersede(self.generation, now_ms);
        }
        #[cfg(target_arch = "wasm32")]
        {
            // No threads in the browser: Apply works, the live preview does not.
            let _ = (ctx, self.want, self.generation, &self.pending, &self.last_requested);
        }
    }

    /// Takes a finished job: the opening analysis when one is pending, else a preview frame that
    /// belongs to the running request and the same revision. Anything older was superseded or
    /// drawn from replaced pixels and is dropped.
    #[cfg(not(target_arch = "wasm32"))]
    fn poll(&mut self, app: &mut PhotocraftApp, ctx: &egui::Context) {
        let Some((key, result)) = self.worker.poll() else { return };
        match result {
            Ok(worker::Output::Analysis(reply)) => {
                if self.analysis_pending.take().is_some() {
                    self.finish(app, reply.anchors, reply.fit, reply.contour, reply.quality, reply.layer);
                    ctx.request_repaint();
                }
            }
            Ok(worker::Output::Frame(frame)) => {
                let current = self.pending.is_some_and(|(k, rev)| k == key && rev == self.revision);
                self.pending = None;
                if current {
                    self.publish(app, frame);
                    ctx.request_repaint();
                }
            }
            Err(e) if e != worker::SUPERSEDED => {
                if self.analysis_pending.is_some() {
                    self.fail(e);
                } else {
                    self.preview_error = Some(e);
                }
                ctx.request_repaint();
            }
            Err(_) if self.analysis_pending.is_some() => {
                // A cancelled analysis (only a teardown cancels one): leave Apply disabled
                // rather than wait forever for an answer that is not coming.
                self.busy = false;
                self.analysis_pending = None;
                self.notice = Some(tl!("The analysis was cancelled").into());
            }
            Err(_) => {}
        }
    }

    /// Starts the preview for the newest request, unless a previous one is still rendering.
    /// False when the request should be retried next frame.
    #[cfg(not(target_arch = "wasm32"))]
    fn start_job(&mut self, app: &mut PhotocraftApp, ctx: &egui::Context) -> bool {
        if self.worker.busy() {
            return false;
        }
        let Some(params) = self.last_requested.clone() else { return true };
        let Some(job) = self.job(app, &params) else { return true };
        match self.worker.start(self.generation, ctx.clone(), move |ctx| worker::render(&job, ctx)) {
            Ok(true) => {
                self.pending = Some((self.generation, self.revision));
                true
            }
            Ok(false) => false,
            Err(e) => {
                self.preview_error = Some(e);
                true
            }
        }
    }

    /// The job for `params`: the params with the analysed anchors (what Apply sends), the
    /// document's extent and the proxy with the selection's coverage. `None` when the target
    /// cannot be previewed; why is then in `preview_error`.
    #[cfg(not(target_arch = "wasm32"))]
    fn job(&mut self, app: &mut PhotocraftApp, params: &Value) -> Option<worker::Job> {
        let (proxy, selection) = match self.proxy_for(app) {
            Ok(found) => found,
            Err(why) => {
                self.preview_error = Some(why.to_owned());
                self.suspend();
                return None;
            }
        };
        Some(worker::Job { params: self.merged_params(params), canvas: self.canvas(), face: self.command == super::FACE, proxy, selection })
    }

    /// `params` with the analysed anchors, the body fit, the person contour and the bound
    /// layer — the params Apply sends.
    fn merged_params(&self, params: &Value) -> Value {
        let mut merged = params.clone();
        if let Value::Object(map) = &mut merged {
            map.insert("anchors".into(), self.anchors.clone());
            if let Some(fit) = &self.fit {
                map.insert("fit".into(), fit.clone());
            }
            if let Some(contour) = &self.contour {
                map.insert("contour".into(), contour.clone());
            }
            if let Some(layer) = self.layer {
                map.insert("layer".into(), json!(layer.0));
            }
        }
        merged
    }

    /// The target layer's proxy at the current revision and the selection's coverage at its
    /// pixels, cached until the revision moves.
    #[cfg(not(target_arch = "wasm32"))]
    fn proxy_for(&mut self, app: &mut PhotocraftApp) -> Result<PreviewSource, &'static str> {
        let Some(st) = app.session.active() else { return Err("there is no document to preview") };
        if let Some(cached) = &self.proxy
            && cached.revision == st.revision
        {
            return Ok((Arc::clone(&cached.proxy), cached.selection.clone()));
        }
        let Ok((layer, surface, _)) = crate::distort_ui::active_pixels(app) else {
            return Err("The target layer has no pixels of its own to preview");
        };
        if Some(layer) != self.layer {
            return Err("The live preview can only follow the target layer");
        }
        let proxy = Arc::new(ProxyImage::new(&surface, st.doc.bounds(), PROXY_SIDE));
        let selection = st.doc.selection.as_ref().map(|sel| Arc::new(coverage(sel, &proxy)));
        self.proxy = Some(CachedProxy { revision: st.revision, proxy: Arc::clone(&proxy), selection: selection.clone() });
        Ok((proxy, selection))
    }

    /// Shows `frame`: the layer hidden from the canvas composite and the warped proxy drawn in
    /// its place.
    #[cfg(not(target_arch = "wasm32"))]
    fn publish(&mut self, app: &mut PhotocraftApp, frame: worker::Frame) {
        let (Some(doc_id), Some(layer)) = (self.doc, self.layer) else { return };
        let Some(st) = app.session.active() else { return };
        if st.doc.id != doc_id {
            return;
        }
        let Some(l) = st.doc.layer(layer) else { return };
        let opacity = (l.opacity * l.fill_opacity).clamp(0.0, 1.0);
        let without = crate::distort_ui::without_layer(&st.doc, layer);
        let textures = PreviewTextures::new(format!("sculpt-{}", self.serial), PreviewPixels::new(frame.image, [1.0, 1.0]));
        self.preview = Some(Preview { textures, without, opacity, size: self.size });
    }
}

/// The selection's coverage at each proxy pixel, sampled at the pixel centres as Liquify's
/// preview does; the commit blends at document resolution.
#[cfg(not(target_arch = "wasm32"))]
fn coverage(sel: &Surface, proxy: &ProxyImage) -> Vec<f32> {
    let (bx, by, s) = (f64::from(proxy.bounds.x0), f64::from(proxy.bounds.y0), proxy.scale);
    let mut cov = Vec::with_capacity(proxy.w * proxy.h);
    for y in 0..proxy.h {
        let dy = (by + (y as f64 + 0.5) * s).floor() as i32;
        for x in 0..proxy.w {
            let dx = (bx + (x as f64 + 0.5) * s).floor() as i32;
            cov.push(sel.sample_channel(dx, dy, 0).clamp(0.0, 1.0));
        }
    }
    cov
}

/// Why a live preview would show a warp the composite would not, if it would.
fn suppression(app: &PhotocraftApp, doc: &Document, layer_id: LayerId) -> Option<&'static str> {
    let layer = doc.layer(layer_id)?;
    if !layer.visible {
        return Some("the target layer is hidden, so the live preview would show it again");
    }
    if layer.mask.as_ref().is_some_and(|mask| mask.enabled && !reveals_all(mask)) {
        return Some("the target layer's mask shapes its pixels, which the live preview cannot show");
    }
    if layer.vector_mask.as_ref().is_some_and(|mask| mask.enabled && mask.density > 0.0) {
        return Some("the target layer's vector mask shapes its pixels, which the live preview cannot show");
    }
    if layer.clipped {
        return Some("the target layer is clipped to the layer below it, which the live preview cannot show");
    }
    if layer.blend != BlendMode::Normal {
        return Some("the target layer's blend mode changes how it composites, which the live preview cannot show");
    }
    if layer.effects.enabled && !layer.effects.items.is_empty() {
        return Some("the target layer's effects would not show in the live preview");
    }
    if layer.frame.is_some() {
        return Some("the target layer is a Frame, whose clipping the live preview cannot show");
    }
    if !layer.advanced.is_default() || !layer.blend_if.is_default() || layer.excluded_channels != 0 {
        return Some("the target layer's blending options change how it composites, which the live preview cannot show");
    }
    let mut parent = doc.layers.index().parent(layer_id);
    while let Some(id) = parent {
        let Some(group) = doc.layer(id) else { break };
        let artboard = matches!(&group.content, LayerContent::Group(g) if g.artboard.is_some());
        if group.opacity < 1.0
            || group.fill_opacity < 1.0
            || group.blend != BlendMode::PassThrough
            || group.clipped
            || artboard
            || group.frame.is_some()
            || !group.advanced.is_default()
            || !group.blend_if.is_default()
            || group.excluded_channels != 0
            || group.mask.as_ref().is_some_and(|mask| mask.enabled && !reveals_all(mask))
            || group.vector_mask.as_ref().is_some_and(|mask| mask.enabled && mask.density > 0.0)
            || (group.effects.enabled && !group.effects.items.is_empty())
        {
            return Some("the target layer's group changes how it composites, which the live preview cannot show");
        }
        parent = doc.layers.index().parent(id);
    }
    if !crate::distort_ui::colour_set_target(app).is_null() {
        return Some("a colour target is selected, which the live preview cannot show");
    }
    if app.distort.active() || crate::distort_ui::modal_workspace(app).is_some() || app.ui.transform.is_some() {
        return Some("another editing workspace is open, so the live preview is off");
    }
    None
}

/// A mask that is untouched and reveals everything changes nothing: the live preview is exact
/// through it. Density 0 also reveals everything whatever the surface holds.
fn reveals_all(mask: &LayerMask) -> bool {
    mask.density <= 0.0 || (mask.surface.default_pixel().first().is_some_and(|v| *v >= 1.0) && mask.surface.content_bounds().is_empty())
}

/// Applies with the panel's slider values and the analysed anchors: one history step through the
/// engine command. The preview goes first so a frame cannot land on the already-warped pixels.
pub(crate) fn apply(app: &mut PhotocraftApp, panel: &Panel) {
    let Some(mut live) = app.sculpt_live.take() else { return };
    if !live.ready() {
        app.sculpt_live = Some(live);
        return;
    }
    let params = live.merged_params(&panel.params());
    let values = panel.params();
    live.suspend();
    match app.run(live.command, params) {
        Ok(reply) => {
            // A part the pose could not place is not a success to hide: the reply names it,
            // and the panel says so inline (the CLI and automation read `parts` themselves).
            live.notice = parts_notice(&reply);
            // Unchanged values never re-preview: the canvas already shows the committed warp.
            live.last_requested = Some(values);
            #[cfg(not(target_arch = "wasm32"))]
            {
                live.proxy = None;
            }
            live.preview_error = None;
            if let Some(st) = app.session.active() {
                live.revision = st.revision;
            }
        }
        Err(e) => {
            crate::dialogs::alerts::report_error(app, e);
            // The commit changed nothing: let the next frame draw the preview again.
            live.last_requested = None;
        }
    }
    app.sculpt_live = Some(live);
}

/// What an Apply's `parts` report means for the panel's inline status: a region left out or
/// placed by inference is said, not hidden. `None` when every asked region ran on landmarks
/// the pose saw.
fn parts_notice(reply: &Value) -> Option<String> {
    let any = |key: &str| reply.get("parts").and_then(|p| p.get(key)).and_then(Value::as_array).is_some_and(|a| !a.is_empty());
    if any("skipped") {
        Some(tl!("Some regions were left out: the pose did not find their landmarks in the person silhouette").into())
    } else if any("approximate") {
        // Inferred landmarks, a pose too asymmetric to align, a contour the mask fuses with
        // the background: the reasons differ, and the panel names the common truth rather
        // than claiming one of them for all.
        Some(tl!("Some regions were handled approximately: the photo does not show them clearly enough to follow exactly").into())
    } else {
        None
    }
}

/// The document shown on the canvas while a preview hides its target layer.
pub(crate) fn display_doc(app: &PhotocraftApp, idx: usize) -> Option<(Arc<Document>, u64)> {
    let live = app.sculpt_live.as_ref()?;
    if app.session.active_index() != Some(idx) {
        return None;
    }
    let st = app.session.documents().get(idx)?;
    if Some(st.doc.id) != live.doc {
        return None;
    }
    let preview = live.preview.as_ref()?;
    Some((Arc::clone(&preview.without), DISPLAY_KEY + live.serial))
}

/// The warped proxy over the document's rectangle, as a canvas control.
pub(crate) fn show_in_canvas(app: &PhotocraftApp, ui: &mut egui::Ui, rect: egui::Rect) {
    let Some(live) = &app.sculpt_live else { return };
    let Some(preview) = &live.preview else { return };
    if app.session.active().map(|st| st.doc.id) != live.doc {
        return;
    }
    let (w, h) = (f64::from(preview.size[0].max(1)), f64::from(preview.size[1].max(1)));
    let quad = [Point::new(0.0, 0.0), Point::new(w, 0.0), Point::new(w, h), Point::new(0.0, h)];
    TransformPreview::new(tl!("Sculpt preview"), &preview.textures).map(PreviewMap::quad(quad)).opacity(preview.opacity).clip(preview.size).show_at(ui, rect);
}

#[cfg(test)]
impl Live {
    /// A panel runtime with the anchors an analysis would have found, without running the models.
    pub(crate) fn test(app: &mut PhotocraftApp, command: &'static str, anchors: Value) -> Live {
        let serial = app.ui.alloc_id();
        let st = app.session.active().expect("a test document");
        Live {
            command,
            serial,
            doc: Some(st.doc.id),
            layer: st.active_layer,
            size: [st.doc.size.width, st.doc.size.height],
            revision: st.revision,
            anchors,
            fit: None,
            contour: None,
            quality: None,
            roi: None,
            sampled: true,
            warpable: true,
            busy: false,
            notice: None,
            preview_error: None,
            suppress: None,
            last_requested: Some(json!({})),
            generation: 0,
            want: false,
            pending: None,
            preview: None,
            #[cfg(not(target_arch = "wasm32"))]
            proxy: None,
            #[cfg(not(target_arch = "wasm32"))]
            analysis_pending: None,
            #[cfg(not(target_arch = "wasm32"))]
            worker: worker::Worker::default(),
        }
    }

    /// Runs the opening analysis on this thread, through the same engine call and finish step
    /// the worker's answer takes, so a test needs no timing.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn analysis_now(&mut self, app: &mut PhotocraftApp) {
        // Cancel the worker's copy first: this runs the same read-only analysis again.
        self.worker.cancel();
        let Some(layer) = self.layer else { return };
        let Some(st) = app.session.active() else { return };
        if Some(st.doc.id) != self.doc {
            return;
        }
        let job = worker::AnalysisJob { doc: Arc::clone(&st.doc), layer, kind: self.kind(), roi: self.roi };
        match worker::analyze(&job, &photocraft_engine::jobs::JobCtx::new()) {
            Ok(worker::Output::Analysis(reply)) => self.finish(app, reply.anchors, reply.fit, reply.contour, reply.quality, reply.layer),
            Ok(worker::Output::Frame(_)) => {}
            Err(e) => self.fail(e),
        }
    }

    /// Whether a frame is shown: the canvas hides the target layer and draws the proxy.
    pub(crate) fn has_preview(&self) -> bool {
        self.preview.is_some()
    }

    /// Renders and publishes one frame on this thread, so the test needs no timing; the frame
    /// is returned to compare pixels.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn render_now(&mut self, app: &mut PhotocraftApp, panel: &Panel) -> Option<egui::ColorImage> {
        let params = panel.params();
        self.last_requested = Some(params.clone());
        let job = self.job(app, &params)?;
        let frame = match worker::render(&job, &photocraft_engine::jobs::JobCtx::new()).ok()? {
            worker::Output::Frame(frame) => frame,
            worker::Output::Analysis(_) => return None,
        };
        let image = frame.image.clone();
        self.publish(app, frame);
        Some(image)
    }
}

#[cfg(test)]
mod tests;
