clearvars; 

estimulo = csvread("estimulos_completos.csv"); 
[ensayos_totales, atributos] = size(estimulo);
idx = [1:ensayos_totales];

mat_w = zeros(10, 10);
[som_high, som_length] = size(mat_w);
neuronas = som_high.*som_length;
som = rand(neuronas, atributos)*.50;

cellSOM = cell(som_high, som_length); 
alpha = 0.7;
sigm = 6;

l = 0; 
    while l < som_high
    l = l + 1;
    h = 0;
    while h < som_length
        h = h + 1;
        cellSOM(h,l) = {[h,l]};
    end
    end
    
cellSOM_column = reshape(cellSOM, neuronas, 1);
matrix_SOM = cell2mat(cellSOM_column); 

T = 500; 
for t=1:T 

%Descenso de Alpha
alpha_loop = alpha*(1-(t/T));

if alpha_loop < 0.1
    alpha_loop = 0.1; 
else
    alpha_loop;
end

%Descenso de Sigma 
sigm_loop = sigm*(1-(t/T));

if  sigm_loop < 1 
    sigm_loop = 1;
else
    sigm_loop;
end

%Shuffle estimation matrix
estim_idx_sort = randperm(ensayos_totales);
%matrix_estim_rand = estimulo(randperm(size(estimulo, 1)), :); 

for i=1:ensayos_totales

estimulo_loop = estimulo(estim_idx_sort(i),:);
store_estimulo_loop(i,:) = estimulo_loop;
%Calculando las diferencias del som
diferencias = pdist2(som, estimulo_loop, 'euclidean'); 
diferencias_matrix = reshape(diferencias, som_high,som_length,1); 
%Calculando a la ganadora
min_value1 = min(min(abs(diferencias_matrix)));  
[x, y] = find(diferencias_matrix == min_value1); 
win = [x, y];
win_store(estim_idx_sort(i),:) = win;
win_colum = repmat(win, neuronas, 1);  

%Calculando las distancias
distancias = pdist2(matrix_SOM, win_colum,'euclidean'); 
distancias = distancias(:,1); 
distancias_matrix = reshape(distancias, som_high,som_length); 

%Calculamos la función de vecindad gaussiana y actualización del som
distancias_norm_gauss = exp(-((distancias.^2)/(2*(sigm_loop)^2)));
error = estimulo_loop - som; 
som = som + (alpha_loop .* distancias_norm_gauss .* error);

%Calculando la activacion del som
activacion_som = exp(-(diferencias_matrix)./max(exp(-(diferencias_matrix))));

end

 store_estimulo_loop_t(:, :, t) = store_estimulo_loop;  
 win_store_t(:,:, t) = win_store; 
 
end

% win_final = win_store;
% estimulo_final = store_estimulo_loop;

%%FIN 