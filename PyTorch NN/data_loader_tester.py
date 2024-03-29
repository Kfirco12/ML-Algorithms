from gcommand_loader import GCommandLoader
import torch

#dataset = GCommandLoader('./gcommands/test')
dataset = GCommandLoader(r'C:\Users\cohenkfi\Desktop\ML_EX_4\data\data')

test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

for k, (input,label) in enumerate(test_loader):
    print(input.size(), len(label))
