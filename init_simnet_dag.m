function nn = init_simnet()
% INIT_SIMNET  Initialize the Similarity Network

nn = dagnn.DagNN();


dist_fc1 = dagnn.Conv('size',[1 512 2 4096],'pad',0,'stride',1,'hasBias',true);
nn.addLayer('dist_fc_1', dist_fc1, {'rmac_pair'}, {'x1'}, {'fc1f','fc1b'});
nn.addLayer('dist_relu_1', dagnn.ReLU(), {'x1'}, {'x2'});
nn.addLayer('dropout', dagnn.DropOut('rate',0),{'x2'},{'x2d'});
 
dist_fc2 = dagnn.Conv('size',[1 1 4096 4096],'pad',0,'stride',1,'hasBias',true);
nn.addLayer('dist_fc_2', dist_fc2, {'x2d'}, {'x3'}, {'fc2f','fc2b'});
nn.addLayer('dist_relu_2', dagnn.ReLU(), {'x3'}, {'x4'});

dist_fc3 = dagnn.Conv('size',[1 1 4096 1],'pad',0,'stride',1,'hasBias',true);
nn.addLayer('dist_pred', dist_fc3, {'x4'}, {'pred'}, {'fc3f','fc3b'});

nn.addLayer('l1_loss', dagnn.LossRegul(), {'pred', 'label'}, {'objective'});
 
nn.initParams();

end