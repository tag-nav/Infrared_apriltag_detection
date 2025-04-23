# Infrared_marker_detection

## Prerequisites
- python 3.9

## Install packages
```
pip install -r requirements.txt
```

## Install the modified DeepArUco 

The original DeepArUco method: [DeepArUco](https://github.com/AVAuco/deeparuco)

We use the AprilTag flexible layout TagCustom52h12 for our infrared markers. The custom layout can be generated using the [multiscale-marker-generation](https://github.com/tag-nav/multiscale-marker-generation) repository. To enable detection of this layout, we adapted the DeepArUco method accordingly.

Clone the forked DeepArUco repository.
```
git clone https://github.com/tag-nav/infrared_marker_detection.git
cd infrared_marker_detection
git clone https://github.com/tag-nav/deeparuco.git
```

## Generate synthetic images

### Download images

Download [Open Images](https://storage.googleapis.com/openimages/web/index.html) and [FLIR ADAS dataset](https://www.flir.com/oem/adas/adas-dataset-form/?srsltid=AfmBOoqmLHpAgyLD87RJslGMU-ENDrmMRkZ9fRjXPZD5JcmeCtyORtMN).

### Overlay markers onto the downloaded images

To generate synthetic dataset, run the notebook [gen_syn_data.ipynb](https://github.com/tag-nav/infrared_marker_detection/blob/main/gen_syn_img/gen_syn_data.ipynb) to generate synthetic images. Parameters for noise effects and marker placement can be configured in [config_img_with_tag.json](https://github.com/tag-nav/infrared_marker_detection/blob/main/gen_syn_img/config_img_with_tag.json).

