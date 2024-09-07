# This project libraries:
from _func import *

# Files and folders addresses:
root = str(Path(__file__).parent) + "\\"
input_addr = root + 'inputs.txt'
rm_out_addr = root + 'outputs_rm.txt'
dm_out_addr = root + 'outputs_dm.txt'
ed_out_addr = root + 'outputs_ed.txt'
ap_out_addr = root + 'outputs_ap.txt'
input_addr2 = root + 'inputs-resource.txt'
rm_res_out_addr = root + 'outputs_rm_resource.txt'
rm_res_inheritance_out_addr = root + 'outputs_rm_resource_inheritance.txt'
imgs_folder_addr = root + 'imgs\\'
Path(imgs_folder_addr).mkdir(parents=True, exist_ok=True)

# Reading Examples
examples = file_reader(input_addr)
examples2 = file_reader(input_addr2)

# Scheduling:
rm_results = rm_scheduler(examples)
dm_results = dm_scheduler(examples)
ed_results = ed_scheduler(examples)
interupt_time, interupt_job = 3, 8
ap_results = ap_rm_scheduler(examples, interupt_time, interupt_job)
rm_resource_inheritance_result = resourced_rm_with_inheritance(examples2)
rm_resource_result = resourced_rm_without_inheritance(examples2)

# Writing the results in files:
file_writer(rm_results, rm_out_addr)
file_writer(dm_results, dm_out_addr)
file_writer(ed_results, ed_out_addr)
file_writer(ap_results, ap_out_addr)
file_writer2(rm_resource_result, rm_res_out_addr)
file_writer2(rm_resource_inheritance_result, rm_res_inheritance_out_addr)

# Saving figures (Optional)
save_figs(examples, rm_results, "RM", imgs_folder_addr)
save_figs(examples, dm_results, "DM", imgs_folder_addr)
save_figs(examples, ed_results, "ED", imgs_folder_addr)
save_figs(examples, ap_results, "AP", imgs_folder_addr)
save_figs2(examples2, rm_resource_inheritance_result, "RM with resource", imgs_folder_addr)
save_figs2(examples2, rm_resource_result, "RM with resource with priority inheritance", imgs_folder_addr)