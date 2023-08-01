function [cellSpatialData] = generateSpatialData9x9(data)
sz = size(data);

data = [ data([1 2 3 4],:,:); data ; data([end-3 end-2 end-1 end],:,:)];

data = [  data(:,[1 2 3 4],:) data  data(:,[end-3 end-2 end-1 end],:) ];

for i = 5:sz(1)+4
    for j = 5:sz(2)+4
        cellSpatialData{i-4,j-4} = [ 
            reshape(data(i-4,j-4,:),[1 sz(3)]) 
            reshape(data(i-4,j-3,:),[1 sz(3)]) 
            reshape(data(i-4,j-2,:),[1 sz(3)]) 
            reshape(data(i-4,j-1,:),[1 sz(3)]) 
            reshape(data(i-4,j,:),[1 sz(3)]) 
            reshape(data(i-4,j+1,:),[1 sz(3)]) 
            reshape(data(i-4,j+2,:),[1 sz(3)])
            reshape(data(i-4,j+3,:),[1 sz(3)])  
	    reshape(data(i-4,j+4,:),[1 sz(3)])    
        
            reshape(data(i-3,j-4,:),[1 sz(3)])           
            reshape(data(i-3,j-3,:),[1 sz(3)]) 
            reshape(data(i-3,j-2,:),[1 sz(3)]) 
            reshape(data(i-3,j-1,:),[1 sz(3)]) 
            reshape(data(i-3,j,:),[1 sz(3)]) 
            reshape(data(i-3,j+1,:),[1 sz(3)]) 
            reshape(data(i-3,j+2,:),[1 sz(3)])
            reshape(data(i-3,j+3,:),[1 sz(3)])
            reshape(data(i-3,j+4,:),[1 sz(3)])
            
            reshape(data(i-2,j-4,:),[1 sz(3)])                       
            reshape(data(i-2,j-3,:),[1 sz(3)]) 
            reshape(data(i-2,j-2,:),[1 sz(3)]) 
            reshape(data(i-2,j-1,:),[1 sz(3)]) 
            reshape(data(i-2,j,:),[1 sz(3)]) 
            reshape(data(i-2,j+1,:),[1 sz(3)]) 
            reshape(data(i-2,j+2,:),[1 sz(3)])
            reshape(data(i-2,j+3,:),[1 sz(3)])
            reshape(data(i-2,j+4,:),[1 sz(3)])           

            reshape(data(i-1,j-4,:),[1 sz(3)])                                   
            reshape(data(i-1,j-3,:),[1 sz(3)]) 
            reshape(data(i-1,j-2,:),[1 sz(3)]) 
            reshape(data(i-1,j-1,:),[1 sz(3)]) 
            reshape(data(i-1,j,:),[1 sz(3)]) 
            reshape(data(i-1,j+1,:),[1 sz(3)]) 
            reshape(data(i-1,j+2,:),[1 sz(3)])
            reshape(data(i-1,j+3,:),[1 sz(3)])
            reshape(data(i-1,j+4,:),[1 sz(3)])           
       
            reshape(data(i,j-4,:),[1 sz(3)])                                              
            reshape(data(i,j-3,:),[1 sz(3)]) 
            reshape(data(i,j-2,:),[1 sz(3)]) 
            reshape(data(i,j-1,:),[1 sz(3)]) 
            reshape(data(i,j,:),[1 sz(3)]) 
            reshape(data(i,j+1,:),[1 sz(3)]) 
            reshape(data(i,j+2,:),[1 sz(3)])
            reshape(data(i,j+3,:),[1 sz(3)])
            reshape(data(i,j+4,:),[1 sz(3)])           
            
            reshape(data(i+1,j-4,:),[1 sz(3)])                                              
            reshape(data(i+1,j-3,:),[1 sz(3)]) 
            reshape(data(i+1,j-2,:),[1 sz(3)]) 
            reshape(data(i+1,j-1,:),[1 sz(3)]) 
            reshape(data(i+1,j,:),[1 sz(3)]) 
            reshape(data(i+1,j+1,:),[1 sz(3)]) 
            reshape(data(i+1,j+2,:),[1 sz(3)])
            reshape(data(i+1,j+3,:),[1 sz(3)])
            reshape(data(i+1,j+4,:),[1 sz(3)])           
            
            reshape(data(i+2,j-4,:),[1 sz(3)])                                              
            reshape(data(i+2,j-3,:),[1 sz(3)]) 
            reshape(data(i+2,j-2,:),[1 sz(3)]) 
            reshape(data(i+2,j-1,:),[1 sz(3)]) 
            reshape(data(i+2,j,:),[1 sz(3)]) 
            reshape(data(i+2,j+1,:),[1 sz(3)]) 
            reshape(data(i+2,j+2,:),[1 sz(3)])
            reshape(data(i+2,j+3,:),[1 sz(3)])
            reshape(data(i+2,j+4,:),[1 sz(3)])           
           
            reshape(data(i+3,j-4,:),[1 sz(3)])                                              
            reshape(data(i+3,j-3,:),[1 sz(3)]) 
            reshape(data(i+3,j-2,:),[1 sz(3)]) 
            reshape(data(i+3,j-1,:),[1 sz(3)]) 
            reshape(data(i+3,j,:),[1 sz(3)]) 
            reshape(data(i+3,j+1,:),[1 sz(3)]) 
            reshape(data(i+3,j+2,:),[1 sz(3)])
            reshape(data(i+3,j+3,:),[1 sz(3)])
            reshape(data(i+3,j+4,:),[1 sz(3)])           

            reshape(data(i+4,j-4,:),[1 sz(3)]) 
            reshape(data(i+4,j-3,:),[1 sz(3)]) 
            reshape(data(i+4,j-2,:),[1 sz(3)]) 
            reshape(data(i+4,j-1,:),[1 sz(3)]) 
            reshape(data(i+4,j,:),[1 sz(3)]) 
            reshape(data(i+4,j+1,:),[1 sz(3)]) 
            reshape(data(i+4,j+2,:),[1 sz(3)])
            reshape(data(i+4,j+3,:),[1 sz(3)])  
	    reshape(data(i+4,j+4,:),[1 sz(3)])    
 ];    
           
    end 
end

% plot(cellSpatialData{3,2}(1,:))

