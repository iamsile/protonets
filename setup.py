from setuptools import setup

setup(name='protonets',
      version='0.0.1',
      author='Angelos Filos',
      author_email='filos.angel@gmail.com',
      license='MIT',
      packages=['protonets'],
      install_requires=[
          'tensorflow',
          'tqdm',
          'pyyaml',
          'Pillow'
      ],
      extras_require={
          'dev': [
              'matplotlib',
              'jupyterlab',
              'pylint',
              'autopep8'
          ]
      })
