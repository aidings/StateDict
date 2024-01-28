#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 Vivo Inc. All Rights Reserved.
# @Time             : 2019/10/20 8:59 AM
# @Author           : ding
# @File             : setup.py
# @description      :
from setuptools import find_packages, setup

setup(name='StateDict',  # 包名
      version='0.0.1',  # 版本号
      description='load state dict into model',
      long_description='',
      author='zhifeng.ding',
      author_email='zhifeng.ding@vivo.com',
      url='',
      license='',
      install_requires=['torch', 'safetensors', 'loguru'],
      extras_require={},
      dependency_links=[
          "https://pypi.tuna.tsinghua.edu.cn/simple",
          "http://mirrors.aliyun.com/pypi/simple"
      ],
      classifiers=[
          'Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'Natural Language :: Chinese (Simplified)',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.2'
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Topic :: Utilities'
      ],
      keywords='',
      packages=find_packages('src', exclude=["examples", "tests", "project"]),  # 必填
      package_dir={'': 'src'},  # 必填
      include_package_data=True,
      scripts= [
          # './script/xxx.py'
      ],
)