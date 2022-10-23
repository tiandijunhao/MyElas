try:
    from setuptools import setup, find_packages
    use_setuptools = True
    print('setuptools is used')
except ImportError:
    from distutils.core import setup, Extension
    use_setuptools = False
    print('distutils is used')

setup(
    name='Myelas',
    packages=["myelas"],
    install_requires=['numpy', 'spglib', 'matplotlib'],
    author="Hao Wang",
    author_email="wh_95@qq.com",
    scripts=['scripts/Myelas'],
    zip_safe=False,
    license="LICENSE.txt"
)
