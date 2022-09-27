
from image_threshold import ImageThreshold


def run_baboon():
  baboon_threshold = ImageThreshold('images/baboon.pgm')
  baboon_threshold.global_thresholding()
  baboon_threshold.bernsen()
  baboon_threshold.niblack(1)
  baboon_threshold.sauvola_pietaksinen()
  baboon_threshold.phansalskar(5)
  baboon_threshold.contrast()
  baboon_threshold.mean()
  baboon_threshold.median()

def run_fiducial():
  fiducial_threshold = ImageThreshold('images/fiducial.pgm')
  fiducial_threshold.global_thresholding()
  fiducial_threshold.bernsen()
  fiducial_threshold.niblack(1)
  fiducial_threshold.sauvola_pietaksinen()
  fiducial_threshold.phansalskar(5)
  fiducial_threshold.contrast()
  fiducial_threshold.mean()
  fiducial_threshold.median()

def run_monarch():
  monarch_threshold = ImageThreshold('images/monarch.pgm')
  monarch_threshold.global_thresholding()
  monarch_threshold.bernsen()
  monarch_threshold.niblack(1)
  monarch_threshold.sauvola_pietaksinen()
  monarch_threshold.phansalskar(5)
  monarch_threshold.contrast()
  monarch_threshold.mean()
  monarch_threshold.median()

def run_peppers():
  peppers_threshold = ImageThreshold('images/peppers.pgm')
  peppers_threshold.global_thresholding()
  peppers_threshold.bernsen()
  peppers_threshold.niblack(1)
  peppers_threshold.sauvola_pietaksinen()
  peppers_threshold.phansalskar(5)
  peppers_threshold.contrast()
  peppers_threshold.mean()
  peppers_threshold.median()

def run_retina():
  retina_threshold = ImageThreshold('images/retina.pgm')
  retina_threshold.global_thresholding()
  retina_threshold.bernsen()
  retina_threshold.niblack(1)
  retina_threshold.sauvola_pietaksinen()
  retina_threshold.phansalskar(5)
  retina_threshold.contrast()
  retina_threshold.mean()
  retina_threshold.median()

def run_sonnet():
  sonnet_threshold = ImageThreshold('images/sonnet.pgm')
  sonnet_threshold.global_thresholding()
  sonnet_threshold.bernsen()
  sonnet_threshold.niblack(1)
  sonnet_threshold.sauvola_pietaksinen()
  sonnet_threshold.phansalskar(5)
  sonnet_threshold.contrast()
  sonnet_threshold.mean()
  sonnet_threshold.median()

def run_wedge():
  wedge_threshold = ImageThreshold('images/wedge.pgm')
  wedge_threshold.global_thresholding()
  wedge_threshold.bernsen()
  wedge_threshold.niblack(1)
  wedge_threshold.sauvola_pietaksinen()
  wedge_threshold.phansalskar(5)
  wedge_threshold.contrast()
  wedge_threshold.mean()
  wedge_threshold.median()

run_baboon()
run_fiducial()
run_monarch()
run_peppers()
run_retina()
run_sonnet()
run_wedge()