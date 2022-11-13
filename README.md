# My thesis code repository
Thesis title: Mapping of cloud data from meteorological measurements into observations of Mini-EUSO experiment.

## Table of contents
* [General info](#general-info)
* [Abstract](#abstract)
* [Files](#files)
* [Links](#links)

## General info
This repository contains Python script and Jupyter notebooks that were part of my thesis. There are markdown cells with explanations available in each file.

## Abstract
Mini-EUSO is a smaller prototype version of the main telescope of the JEM-EUSO mission, which was created as one of the preparatory experiments of this mission. It was placed on the Russian module Zvezda of the International Space Station on August 27, 2019, from where this device observes various Earth, atmospheric and cosmic events through a UV-transparent window. One of the most disturbing factors in the accuracy of its measurements is the clouds, which are not perfectly recognizable by the detector itself. The first section of this thesis is dedicated to finding the intersections between Mini-EUSO data and cloud data from various meteorological sources. Subsequently, the cloud data are transformed into the projection of the pixels of the Mini-EUSO detector. The second section of this work covers the creation of a tool for visualizing the transformed data, which is afterward used for its comparison. In addition to the visualizations comparison, the algorithmically calculated point biserial correlation coefficients are also compared, but the resulting values are deemed unusable due to the time difference of data acquisitions and the occurrence of dead pixels. However, on some visualizations, it is possible to see similarities between the cloud data and the airglow values measured by the Mini-EUSO detector.

## Files
#### get_intersections.py
This script was used for finding intersections between available Mini-EUSO data and cloud data available from 4 different projects - MODIS, Planet, Landsat and Sentinel-2.

#### Intersections data analysis.ipynb
Analysis of the pickle files that were the output of the get_intersections.py script.

#### Mini-EUSO Cloud Mapping.ipynb
This notebook contains step by step mapping of downloaded MODIS cloud data from available intersections to Mini-EUSO data.

#### Visualizing best intersections and calculating correlation coefficient.ipynb
This notebook contains all the necessary visualizations and calculations of correlation coefficient.

## Links
* [Full thesis](https://opac.crzp.sk/?fn=detailBiblioForm&sid=83298E4B59163389188EDDB92829&seo=CRZP-detail-kniha) (in Slovak)
* [Mini-EUSO mission](https://www.jemeuso.org/missions/mini-euso/)
* [Mini-EUSO data acquisition](https://www.spiedigitallibrary.org/journals/Journal-of-Astronomical-Telescopes-Instruments-and-Systems/volume-5/issue-4/044009/Mini-EUSO-data-acquisition-and-control-software/10.1117/1.JATIS.5.4.044009.short?SSO=1)
* [MODIS](https://modis.gsfc.nasa.gov/)
