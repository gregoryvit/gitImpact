from setuptools import setup

setup(
    name='gitimpact',
    version='0.2.0',
    packages=['gitimpact', 'gitimpact.issues', 'gitimpact.issues.redmine', 'gitimpact.formatters'],
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
        "gitpython",
        "python-redmine==1.5.1",
        "ruamel.yaml==0.13.14"
    ]
)