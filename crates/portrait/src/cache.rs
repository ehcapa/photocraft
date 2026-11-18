//! The Session-held analysis cache: the SubjectCache pattern (smartselect_cmds), minus the
//! engine internals. [`Cache`] is runtime state, not serialized — it never touches a document
//! file.

use std::sync::{Arc, Weak};

use photocraft_doc::{DocId, Document};

use crate::analysis::Analysis;

/// The portrait pipeline's analysis of the document it last ran on, reused while the pixels
/// are unchanged. `get` validates through a `same_pixels` closure the engine passes in (its
/// tile-pointer comparison), so this crate stays engine-free.
///
/// The skeleton's own materialization always damages the document, so a hit needs a re-run
/// with no intervening edit — the wiring exists because the per-stage invalidation the
/// roadmap builds on ("rerun only the stage whose region changed") lands on this same cache
/// without re-plumbing the Session.
#[derive(Debug, Default)]
pub struct Cache {
    doc: Weak<Document>,
    doc_id: Option<DocId>,
    analysis: Option<Arc<Analysis>>,
}

impl Cache {
    /// The analysis of `current`, when it was computed from these very pixels.
    pub fn get(&self, current: &Document, same_pixels: impl FnOnce(&Document, &Document) -> bool) -> Option<Arc<Analysis>> {
        let cached = self.doc.upgrade()?;
        (self.doc_id == Some(current.id) && same_pixels(&cached, current)).then(|| self.analysis.clone()).flatten()
    }

    /// Records `analysis` as of `current`'s pixels. `current` is the session's live
    /// `Arc<Document>` (the SubjectCache pattern: the Weak points at the allocation the
    /// session holds, never at a copy), so no document is cloned here.
    pub fn store(&mut self, current: &Arc<Document>, analysis: Analysis) {
        self.doc = Arc::downgrade(current);
        self.doc_id = Some(current.id);
        self.analysis = Some(Arc::new(analysis));
    }

    /// Drops the entry when the closed document was the cached one.
    pub fn forget_doc(&mut self, id: DocId) {
        if self.doc_id.is_some_and(|cached| cached == id) {
            *self = Default::default();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use photocraft_doc::PixelFormat;
    use photocraft_doc::{ColorMode, Document, SampleType, Size};
    use photocraft_raster::Surface;

    use super::*;
    use crate::recipe::Region;

    fn doc() -> Arc<Document> {
        Arc::new(Document::new("Untitled", Size { width: 8, height: 8 }, ColorMode::Rgb, SampleType::U8))
    }

    fn analysis() -> Analysis {
        let mut a = Analysis::new(8, 8);
        a.set(Region::Person, Surface::with_default(PixelFormat::GRAY8, &[1.0]));
        a
    }

    #[test]
    fn stored_analysis_hits_while_the_pixels_match() {
        let mut cache = Cache::default();
        let d = doc();
        cache.store(&d, analysis());
        let hit = cache.get(&d, |_, _| true).unwrap();
        assert_eq!(hit.width, 8);
    }

    #[test]
    fn a_miss_for_changed_pixels_another_document_or_a_dropped_one() {
        let mut cache = Cache::default();
        let d = doc();
        cache.store(&d, analysis());
        // The engine's tile comparison says the pixels changed.
        assert!(cache.get(&d, |_, _| false).is_none());
        // Another document id: not ours, even with matching pixels.
        let other = Arc::new(Document::new("Other", Size { width: 8, height: 8 }, ColorMode::Rgb, SampleType::U8));
        assert!(cache.get(&other, |_, _| true).is_none());
        // The cached document dropped: nothing to compare against.
        drop(d);
        let fresh = doc();
        assert!(cache.get(&fresh, |_, _| true).is_none());
    }

    #[test]
    fn forget_doc_clears_only_its_own_document() {
        let mut cache = Cache::default();
        let d = doc();
        cache.store(&d, analysis());
        cache.forget_doc(DocId(d.id.0 + 1));
        assert!(cache.get(&d, |_, _| true).is_some(), "another document's close keeps the entry");
        cache.forget_doc(d.id);
        assert!(cache.get(&d, |_, _| true).is_none());
    }

    #[test]
    fn store_points_at_the_callers_arc_not_a_copy() {
        // The Weak must reference the caller's allocation, not a private copy of the
        // document: a copy would both cost a full clone and never compare equal by tiles.
        let mut cache = Cache::default();
        let d = doc();
        cache.store(&d, analysis());
        drop(d);
        // With the caller's Arc gone the upgrade fails — no second copy is keeping the
        // document alive inside the cache.
        let fresh = doc();
        assert!(cache.get(&fresh, |_, _| true).is_none());
    }
}
