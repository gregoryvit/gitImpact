from setuptools import setup

setup(
    name='gitimpact',
    version='0.6.0',
    packages=['gitimpact', 'gitimpact.core',
              'gitimpact.core.calc', 'gitimpact.core.addons'],
    url='',
    license='MIT',
    author='gregoryvit',
    author_email='gregoryvit@gmail.com',
    description='Automatic git impact analysis tool',
    scripts=[
        'bin/gitimpact'
    ],
    package_data={
        'gitimpact': ['templates/*']
    },
    install_requires=[
        "gitpython==3.1.32",
        "python-redmine==1.5.1",
        "ruamel.yaml==0.13.14",
        "sklearn==0.0",
        "pandas==0.23.4",
        "numpy==1.15.2",
        "reprint==0.5.1",
        "futures==3.2.0"
    ]
)
