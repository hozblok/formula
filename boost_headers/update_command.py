# Use Python 3.10.
# Script to precollect boost headers in the 'boost' catalog.
# We don't need to update boost headers very often. So we can bootstrap all headers here
# in the 'boost' catalog and use it everywhere.
import argparse
import urllib.request
import os
from pathlib import Path
import shutil
import zipfile

parser = argparse.ArgumentParser(description="Download and update Boost headers.")
parser.add_argument("--version", required=True, help="version of the Boost e.g. 1.79.0")
args = parser.parse_args()
version: str = args.version
version_list = version.split(".")
if len(version_list) != 3 or any(not el.isdigit() for el in version_list):
    raise Exception("Wrong version, the pattern is D.DD.D, e.g. 1.78.0")

script_dir = Path(__file__).parent

file_name = "_".join(("boost", *version_list))
file_name_ext = file_name + ".zip"
print("*" * 20)
print("Output:", file_name_ext)

print("Download...")

# TODO
# url = f"https://boostorg.jfrog.io/artifactory/main/release/{version}/source/{file_name_ext}"
# chunk_size = 65536
# with urllib.request.urlopen(url) as response, open(script_dir / file_name_ext, "wb") as out_file:
#     while data := response.read(chunk_size):
#         out_file.write(data)

print("Unzip...")

with zipfile.ZipFile(script_dir / file_name_ext, "r") as zip_ref:
    zip_ref.extractall(script_dir)

print("Delete all except the 'boost' catalog with headers...")
boost_entry: os.DirEntry | None = None
for el in os.scandir(script_dir / file_name):
    if el.name != "boost":
        if el.is_dir():
            shutil.rmtree(el)
        else:
            os.remove(el)
    else:
        boost_entry = el

if boost_entry is None:
    raise Exception("Boost catalog does not exist!")

print("Delete unused boost libs...")
allowed_libs = {
    "algorithm",
    "assert",
    "bind",
    "concept",
    "config",
    "container",
    "container_hash",
    "core",
    "detail",
    "exception",
    "format",
    "function",
    "functional",
    "hana",
    "integer",
    "iterator",
    "lambda",
    "lexical_cast",
    "math",
    "move",
    "mpl",
    "multiprecision",
    "numeric",
    "optional",
    "predef",
    "preprocessor",
    "range",
    "smart_ptr",
    "tuple",
    "type_index",
    "type_traits",
    "typeof",
    "utility",
    "variant",
    "winapi",
}
counter_del = counter_ok = 0
for el in os.scandir(boost_entry):
    if el.is_dir():
        if el.name not in allowed_libs:
            try:
                shutil.rmtree(el)
                counter_del += 1
            except PermissionError:
                print(el.path, "cannot be removed!")
        else:
            counter_ok += 1
            allowed_libs.discard(el.name)

if allowed_libs:
    raise Exception("There are no such catalogues:", allowed_libs)
print(f">>> Removed: {counter_del}, left unremoved: {counter_ok}")

print("Move new boost_headers files...")
shutil.rmtree(script_dir / "boost")
Path(boost_entry).rename(script_dir / "boost")

print("Clean up...")
shutil.rmtree(script_dir / file_name)
# TODO
# os.remove(script_dir / file_name_ext)
print("*"*20)