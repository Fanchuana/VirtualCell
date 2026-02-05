import torch

base_path = "/work/home/cryoem666/czx/project/state_verification_02_04/experiment/STATE_02_02/model_output/ST-HVG-Replogle"
toml_path = "/work/home/cryoem666/czx/project/state_verification_02_04/config"

for task in ['fewshot', 'zeroshot']:
    for name in ['hepg2', 'jurkat', 'k562', 'rpe1']:
        torch_path = f"{base_path}/{task}/{name}/data_module.torch"

        file = torch.load(torch_path)
        file['toml_config_path'] = f"{toml_path}/{task}_{name}.toml"

        torch.save(file, torch_path)