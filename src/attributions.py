# Integrated gradients applied to malware programs

import numpy as np
import torch
import torch.nn as nn
from src.model import MalConv

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,# 积分梯度
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)