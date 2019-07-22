from setuptools import setup

setup(name='custom_estimator',
      version='0.1',
      description='AlizTech job interview question',
      url='http://github.com/karpatika92/custom_estimator',
      author='Karpati Andras',
      author_email='karpatika92@gmail.com',
      license='MIT',
      packages=['custom_estimator'],
      install_requires=[
          'numpy','sklearn'
      ],
      zip_safe=False)
