# setup.py
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
import shutil

PACKAGE_TO_LIB_DIR = {'memory': 'tqpmemory', 'IO': 'zero_copy_reader', 'IO/cache': 'tqpcache',
                      'compression': 'tqpcompress', 'operators/kernels': 'tqpkernels'}

SETUP_DIR = Path(__file__).parent.resolve()
LIB_DIR = SETUP_DIR / 'TQPlib'
BUILD_DIR = SETUP_DIR / 'build'  # or read from env, e.g., os.environ.get("TQP_BUILD_DIR")

class build_py(_build_py):
    def run(self):
        print("--- Staging .so files during build_py ---")
        for build_name, lib_name in PACKAGE_TO_LIB_DIR.items():
            build_subdir = BUILD_DIR / build_name
            package_source_dir = LIB_DIR / lib_name
            package_source_dir.mkdir(parents=True, exist_ok=True)
            (package_source_dir / '__init__.py').touch(exist_ok=True)

            so_files = list(build_subdir.glob('*.so'))
            if not so_files:
                raise FileNotFoundError(f"No '*.so' files found in {build_subdir}")

            for file_path in so_files:
                print(f"  Copying {file_path.name} -> {package_source_dir}")
                shutil.copy(file_path, package_source_dir)

        super().run()

setup(
    name='TQPBackend',
    version='1.0.0',  # bump to force rebuild
    packages=["TQPlib.tqpmemory", "TQPlib.zero_copy_reader", "TQPlib.tqpcompress",
              "TQPlib.tqpkernels", "TQPlib.tqpcache"],
    package_dir={"TQPlib.tqpmemory": "TQPlib/tqpmemory", 
                 "TQPlib.zero_copy_reader": "TQPlib/zero_copy_reader",
                 "TQPlib.tqpcompress": "TQPlib/tqpcompress",
                 "TQPlib.tqpkernels": "TQPlib/tqpkernels",
                 "TQPlib.tqpcache": "TQPlib/tqpcache"},
    include_package_data=True,
    package_data={"TQPlib.tqpmemory": ["*.so"], 
                  "TQPlib.zero_copy_reader": ["*.so"], 
                  "TQPlib.tqpcompress" : ["*.so"],
                  "TQPlib.tqpkernels": ["*.so"],
                  "TQPlib.tqpcache": ["*.so"]},
    zip_safe=False,
    cmdclass={"build_py": build_py},
)