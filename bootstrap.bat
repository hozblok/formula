git submodule init
git submodule update --recursive
CD .\boost
git submodule init
git submodule update --recursive
.\bootstrap.bat
.\b2 headers
TODO

 delete dist build 
& ./venv/Scripts/python.exe -m pip install -e .
& ./venv/Scripts/python.exe setup.py test -a "--verbose"

pytest ./tests


python setup.py sdist bdist bdist_wheel bdist_egg

& 'C:\Program Files\Python36\python.exe' setup.py bdist_wheel bdist_egg
& 'C:\Program Files\Python36\python.exe' -m twine upload .\dist\*