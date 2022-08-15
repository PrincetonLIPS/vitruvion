from sketchgraphs import data as datalib
from img2cad import data_utils
from img2cad.pipeline import prerender_images


def test_render_sketches(sketches):
    for sketch in sketches:
        data_utils.normalize_sketch(sketch)
        image_bytes = prerender_images.render_sketch(sketch)
        assert image_bytes is not None


def test_process_sequence(sketches):
    sketch = sketches[0]
    seq = datalib.sketch_to_sequence(sketch)
    result = prerender_images.process_sketch_sequence(seq, 2)
    assert len(result) == 3
