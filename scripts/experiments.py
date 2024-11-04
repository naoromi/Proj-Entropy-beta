### some quick testing with python interface for transcoding

from transcode import driver_hevc_robust

coding_options = {
    '-y': None,       # always overwrite
    '-ss': '0', 
    '-t': '10', 
    '-c:v': 'hevc_nvenc', 
    '-b:v': '800K',
    '-threads': '4',
    '-preset': 'fast', 
    '-f': 'mp4', 
    '-strict': 'experimental',  # or -strict -2 , yes, -2, means same thing
}


## EXPERIMENT: coding with different bitrates
video_list = "/mnt/hbnas/home/fgan/Proj-Entropy/scripts/transcode/jsons/selected_4400_flattened.json"


### HEVC ###

# 800K #
coding_options["-b:v"] = "800K"
driver_hevc_robust.interface(
    json=video_list,
    output_dir="/mnt/hbnas/home/fgan/Proj-Entropy/dataset/hevc_800K",
    coding_options=[item for kv_pair in coding_options.items() for item in kv_pair if item != None])


# 1000K #
coding_options["-b:v"] = "1000K"
driver_hevc_robust.interface(
    json=video_list,
    output_dir="/mnt/hbnas/home/fgan/Proj-Entropy/dataset/hevc_1000K",
    coding_options=[item for kv_pair in coding_options.items() for item in kv_pair if item != None])


# 1200K #
coding_options["-b:v"] = "1200K"
driver_hevc_robust.interface(
    json=video_list,
    output_dir="/mnt/hbnas/home/fgan/Proj-Entropy/dataset/hevc_1200K",
    coding_options=[item for kv_pair in coding_options.items() for item in kv_pair if item != None])


# 1500K #
coding_options["-b:v"] = "1500K"
driver_hevc_robust.interface(
    json=video_list,
    output_dir="/mnt/hbnas/home/fgan/Proj-Entropy/dataset/hevc_1500K",
    coding_options=[item for kv_pair in coding_options.items() for item in kv_pair if item != None])



### VP9 ###


# 800K #


# 1000K #


# 1200K #


# 1500K #

