#Test Model
from src.models import Hang2020
import torch
import os
import pytest
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def test_conv_module():
    m = Hang2020.conv_module(in_channels=369, filters=32)
    image = torch.randn(20, 369, 11, 11)
    output = m(image)
    assert output.shape == (20,32,11,11)

def test_conv_module_maxpooling():
    m = Hang2020.conv_module(in_channels=32, filters=64, maxpool_kernel=(2,2))
    image = torch.randn(20, 32, 11, 11)
    output = m(image, pool = True)
    assert output.shape == (20,64,5,5)

@pytest.mark.parametrize("conv_dimension",[(20,32,11,11),(20,64,5,5),(20,128,2,2)])
def test_spatial_attention(conv_dimension):
    """Check spectral attention for each convoutional dimension"""
    m = Hang2020.spatial_attention(filters=conv_dimension[1])
    image = torch.randn(conv_dimension)
    attention, scores = m(image)
    
@pytest.mark.parametrize("conv_dimension",[(20,32,11,11),(20,64,5,5),(20,128,2,2)])
def test_spectral_attention(conv_dimension):
    """Check spectral attention for each convoutional dimension"""
    m = Hang2020.spectral_attention(filters=conv_dimension[1])
    image = torch.randn(conv_dimension)
    attention, scores = m(image)
    
def test_spectral_network():
    m = Hang2020.spectral_network(bands=369, classes=10)
    image = torch.randn(20, 369, 11, 11)
    output = m(image)
    assert len(output) == 3
    assert output[0].shape == (20,10)
    
def test_spatial_network():
    m = Hang2020.spatial_network(bands=369, classes=10)
    image = torch.randn(20, 369, 11, 11)
    output = m(image)
    assert len(output) == 3
    assert output[0].shape == (20,10)
    
def test_vanillaCNN_HSI():
    m = Hang2020.vanilla_CNN(bands=369, classes=10)
    image = torch.randn(20, 369, 11, 11)
    output = m(image)
    assert output.shape == (20,10)
    
def test_vanillaCNN_RGB():
    m = Hang2020.vanilla_CNN(bands=3, classes=10)
    image = torch.randn(20, 3, 11, 11)
    output = m(image)
    assert output.shape == (20,10)    
    
def test_Hang2020():
    m = Hang2020.Hang2020(bands=3, classes=10)
    image = torch.randn(20, 3, 11, 11)
    output = m(image)
    assert output.shape == (20,10)    
    
def test_load_from_backbone(tmpdir):
    ten_classes = Hang2020.Hang2020(bands=3, classes=10)
    image = torch.randn(20, 3, 11, 11)
    output = ten_classes(image)    
    assert output.shape == (20,10)  
    torch.save(ten_classes.spectral_network.state_dict(), "{}/state_dict.pt".format(tmpdir))
    
    twenty_classes = Hang2020.load_from_backbone(state_dict="{}/state_dict.pt".format(tmpdir), classes=20, bands=3)
    output = twenty_classes(image)[-1]
    assert output.shape == (20,20)  
    