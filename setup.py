from setuptools import setup, find_packages

setup(name='forecast_covid19_positive_cases', 
      version='0.1', 
      packages=find_packages(include=[
          'config',
          'src',
          'tests'
          ]
    )
)