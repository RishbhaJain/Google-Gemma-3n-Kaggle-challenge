from importlib.metadata import PackageNotFoundError

from gemma_experiment.provenance import capture_runtime_environment


class FakeCudnn:
    @staticmethod
    def version():
        return 90100


class FakeBackends:
    cudnn = FakeCudnn()


class FakeCuda:
    def __init__(self, *, available: bool):
        self.available = available

    def is_available(self):
        return self.available

    @staticmethod
    def get_device_name(index):
        assert index == 0
        return "Test GPU"

    @staticmethod
    def get_device_properties(index):
        assert index == 0
        return type("Properties", (), {"total_memory": 24 * 1024**3})()

    @staticmethod
    def get_device_capability(index):
        assert index == 0
        return 8, 9


class FakeTorch:
    __version__ = "2.7.1"
    version = type("Version", (), {"cuda": "12.6"})()
    backends = FakeBackends()

    def __init__(self, *, cuda_available: bool):
        self.cuda = FakeCuda(available=cuda_available)


def fake_version_lookup(package_name):
    if package_name == "datasets":
        return "3.2.0"
    raise PackageNotFoundError(package_name)


def test_runtime_provenance_records_software_and_gpu_contract():
    environment = capture_runtime_environment(
        FakeTorch(cuda_available=True),
        cuda_required=True,
        package_names=("datasets", "unsloth"),
        version_lookup=fake_version_lookup,
    )

    assert environment["pytorch"] == "2.7.1"
    assert environment["cuda"] == "12.6"
    assert environment["cudnn"] == 90100
    assert environment["cuda_required"] is True
    assert environment["gpu"] == "Test GPU"
    assert environment["gpu_total_memory_gib"] == 24.0
    assert environment["gpu_compute_capability"] == "8.9"
    assert environment["packages"] == {"datasets": "3.2.0", "unsloth": None}
    assert environment["python"]
    assert environment["system"]
    assert environment["machine"]


def test_runtime_provenance_has_explicit_cpu_values():
    environment = capture_runtime_environment(
        FakeTorch(cuda_available=False),
        package_names=("datasets",),
        version_lookup=fake_version_lookup,
    )

    assert "cuda_required" not in environment
    assert environment["gpu"] is None
    assert environment["gpu_total_memory_gib"] is None
    assert environment["gpu_compute_capability"] is None
    assert environment["packages"] == {"datasets": "3.2.0"}
