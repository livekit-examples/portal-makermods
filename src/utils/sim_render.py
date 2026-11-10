"""Offscreen Filament cameras, created, updated and destroyed on the sim worker."""
import numpy as np

RENDERERS = ("filament", "classic")


class FilamentRenderer:
    def __init__(self, model, cameras, *, width, height):
        import mujoco
        # MuJoCo 4.0.0 registers the filament: shader/IBL asset provider here.
        # Importing mjrf alone leaves pbr.filamat unresolved in the shipped wheel.
        from mujoco.studio import native_viewer_cc  # noqa: F401
        from mujoco import mjrf
        from mujoco.rendering.filament import renderer

        self.model = model
        self.context = mjrf.Context(mjrf.ContextConfig())
        self.renderer = renderer.Renderer(self.context)
        self.scene = self.renderer.scene("workbench")
        self.scene.configure_from_model(model)
        self.scene.set_msaa(4)
        self.resources = mjrf.ModelResources(self.context, model)
        self.lights = mjrf.ModelLights(self.scene, self.resources)
        self.objects = mjrf.ModelRenderables(self.scene, self.resources)
        self.buffers = {name: np.empty((height, width, 3), dtype=np.uint8) for name in cameras}
        for name, buffer in self.buffers.items():
            self.renderer.target(name, buffer)

    def render(self, data, scene_camera, options):
        import mujoco
        from mujoco import mjrf
        from mujoco.rendering.filament import renderer

        self.objects.set_options(options)  # hide collision meshes, just as in classic
        self.objects.update(data)
        self.lights.update(data)
        views = []
        for name in self.buffers:
            camera = scene_camera
            if name != "scene_camera":
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                camera.fixedcamid = self.model.camera(name.replace("_wrist_camera", "/wrist_camera")).id
            gl = mujoco.mjv_camera2GLCamera(self.model, data, camera)
            # Preserve the complete calibrated frustum (the convenience converter
            # in 4.0.0 omits width/center/orthographic).
            view_camera = mjrf.Camera(**{field: getattr(gl, field) for field in (
                "pos", "forward", "up", "frustum_bottom", "frustum_top", "frustum_near",
                "frustum_far", "frustum_center", "frustum_width", "orthographic")})
            views.append(renderer.View(scene="workbench", target=name, camera=view_camera,
                                       haze=False, headlight=True, headlight_intensity=12000.))
        # The 4.0.0 native backend accepts only one pixel-read request at a time.
        # Wait for each read without advancing physics between cameras.
        for view in views:
            self.renderer.render([view])
        return {name: buffer.copy() for name, buffer in self.buffers.items()}

    def close(self):
        # Python owns these native handles. Release dependents before their context,
        # on the owning worker, after every readback has completed.
        self.objects = self.lights = None
        self.resources = None
        self.scene = None
        self.renderer = None
        self.context = None
        self.buffers.clear()
