clearvars; 

estimulo = csvread("estimulos.csv"); 
[ensayos_totales, atributos] = size(estimulo);

mat_w = zeros(10, 10);
[som_high, som_length] = size(mat_w);
neuronas = som_high.*som_length;
som = rand(neuronas, atributos);

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

T = 50;
for t=1:T 

X = randperm(numel(estimulo));   
matrix_estim_rand = reshape(estimulo(X), size(estimulo));    
store_estim(:,:,t) = matrix_estim_rand; %Aquí vemos que sí los ordena aleat en cada iteracion

for i=1:ensayos_totales

estimulo_loop = matrix_estim_rand(i,:);
diferencias = pdist2(som, estimulo_loop, 'euclidean'); 
diferencias_matrix = reshape(diferencias, som_high,som_length,1); 
min_value1 = min(min(abs(diferencias_matrix)));  
[x, y] = find(diferencias_matrix == min_value1);
win = [x, y];

win_2 = repmat(win, neuronas, 1);  
distancias = pdist2(matrix_SOM, win_2,'euclidean'); 

distancias = distancias(:,1); 
distancias_matrix = reshape(distancias, som_high,som_length); 

%Calculamos la función de vecindad gaussiana
distancias_norm_gauss = exp(-((distancias.^2)/(2*(sigm)^2))); 

error = estimulo_loop - som; 

som = som + (alpha .* distancias_norm_gauss .* error); 
end

%Descenso de Sigma 
sigm = sigm*(1-(t/T));
if sigm < 1
    sigm = 1;
else
    sigm;
end
store_sigm(t,:) = sigm;
%Descenso de Alpha
alpha = alpha*(1-t/T);
store_alpha(t,:)=alpha;
end
%%FIN 