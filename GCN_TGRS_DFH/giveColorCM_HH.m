function classification_map_rgb=giveColorCM_HH(classTest,m,n)
colorList = [0, 205, 0;
    127, 255, 0; 
    46, 139, 87; 
    0, 139, 0; 
    160, 82, 45; 
    0, 255, 255;
    255, 255, 255; 
    216, 191, 216; 
    255, 0, 0; 
    139, 0, 0; 
    0, 100, 255;
    255, 255, 0; 
    238, 154, 0; 
    85, 26, 139;
    255, 127, 80;
    0,0,0; 
    0,0,0;
    0,0,0];
classification_map_rgb = reshape(colorList(classTest,:),m,n,[]);
end


