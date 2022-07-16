.EXPORT_ALL_VARIABLES:

CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0

cifar100:
	@nohup python main.py --config=config/$(@).yaml > $(@).out 2>&1 &

tiny-imagenet200:
	@nohup python main.py --config=config/$(@).yaml > $(@).out 2>&1 &

imagenet100:
	@nohup python main.py --config=config/$(@).yaml > $(@).out 2>&1 &
