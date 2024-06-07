import onecode
import gempy as gp
import gempy_viewer as gpv
from gempy.core.data.enumerators import ExampleModel

from .examples_generator import generate_example_model


def run():
    model = onecode.dropdown(
        "Model",
        "GRABEN",
        options=[
            "HORIZONTAL_STRAT",
            "ANTICLINE",
            "ONE_FAULT",
            "TWO_AND_A_HALF_D",
            "COMBINATION",
            "ONE_FAULT_GRAVITY",
            "GRABEN",
        ]
    )
    r_h = onecode.slider("Resolution (XY)", 50, min=20, max=200, step=1)
    r_v = onecode.slider("Resolution (Z)", 5, min=5, max=200, step=1)
    resolution = [r_h, r_h, r_v]

    onecode.Logger.info(f"Getting Model {model} at resolution {resolution}")
    geo_model = generate_example_model(
        ExampleModel[model],
        resolution=resolution,
        refinement=onecode.slider("Refinement", 2, min=1, max=8, step=1),
        compute_model=False
    )

    onecode.Logger.info("Computing model")
    gp.compute_model(
        geo_model,
        engine_config= gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            dtype="float64",
        )
    )

    onecode.Logger.info("Exporting 3D View")
    plot = gpv.plot_3d(
        model=geo_model,
        show_surfaces=False,
        show_data=True,
        show_lith=False,
        image=False,
        show=False
    )
    plot.p.export_html(onecode.file_output('plot 3d', f'{model}.html'))
