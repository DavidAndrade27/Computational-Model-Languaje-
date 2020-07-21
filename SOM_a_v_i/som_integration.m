clearvars; 

estimulo_1 = csvread("estimulos_forma.csv"); 
estimulo_2 = csvread("estimulos_color.csv");
[ensayos_totales_1, atributos_f] = size(estimulo_1);
[ensayos_totales_2, atributos_c] = size(estimulo_2);
idx_1 = [1:ensayos_totales_1];
idx_2 = [1:ensayos_totales_2];

coords = 4; % Ahorita lo fijo arbitrariamente

mat_w = zeros(10, 10);
[som_high, som_length] = size(mat_w);
neuronas = som_high.*som_length;
som_1 = rand(neuronas, atributos_f)*.50;
som_2 = rand(neuronas, atributos_c)*.50;
som_3 = rand(neuronas, coords)*.50;

HEBBweight_1 = rand(10, 10)*0.5;
HEBBweight_2 = rand(10, 10)*0.5;

cellSOM = cell(som_high, som_length); 
alpha = 0.7;
sigm = 6;
betha = 0.05;
lambda = 0.5;

l = 0; 
    while l < som_high
    l = l + 1;
    h = 0;
    while h < som_length
        h = h + 1;
        cellSOM(h,l) = {[h,l]};
    end
    end
    
cellSOM_column_1 = reshape(cellSOM, neuronas, 1);
cellSOM_column_2 = reshape(cellSOM, neuronas, 1);
cellSOM_column_3 = reshape(cellSOM, neuronas, 1);
matrix_SOM_1 = cell2mat(cellSOM_column_1); 
matrix_SOM_2 = cell2mat(cellSOM_column_2);
matrix_SOM_3 = cell2mat(cellSOM_column_3);

T = 500; 
ultima_epoca = T(end);
 for t=1:T

%Descenso de Alpha
alpha_loop = alpha*(1-(t/T));

if alpha_loop < 0.1
    alpha_loop = 0.1; 
else
    alpha_loop;
end
alpha_loop_store(t,:) = alpha_loop;

%Descenso de Sigma 
sigm_loop = sigm*(1-(t/T));

if  sigm_loop < 1 
    sigm_loop = 1;
else
    sigm_loop;
end
sigm_loop_store(t,:) = sigm_loop;

%Incremento de betha 
betha(t+1) = betha(t)*1.0009;

  
%Shuffle estimation matrix
estim_idx_sort1 = randperm(ensayos_totales_1);
estim_idx_sort2 = estim_idx_sort1;
estim_idx_sort3 = estim_idx_sort1;


for i=1:ensayos_totales_1
    
%%%%%%%%%%%%%%%%%%%%% SOM1 %%%%%%%%%%%%%%%%%%%%%%

estimulo_loop_1 = estimulo_1(estim_idx_sort1(i),:);
store_estimulo_loop_1(estim_idx_sort1(i),:) = estimulo_loop_1;
%Calculando las diferencias del som
diferencias_1 = pdist2(som_1, estimulo_loop_1, 'euclidean'); 
diferencias_matrix_1 = reshape(diferencias_1, som_high,som_length,1);
%Calculando la activacion del som1
act_som1 = exp(-(diferencias_matrix_1)./max(exp(-(diferencias_matrix_1))));
%Calculando BMU1 (Best Matching Unit)y función de vecindad
min_value1 = min(min((diferencias_matrix_1)));  
[x1, y1] = find(diferencias_matrix_1 == min_value1); 
BMU_1 = [x1, y1];
BMU1_store(estim_idx_sort1(i),:) = BMU_1;
BMU1_coord = BMU1_store; %Almacena las win-coordinates1
BMU1_column = repmat(BMU_1, neuronas, 1); 
distancias_BMU1 = pdist2(matrix_SOM_1, BMU1_column,'euclidean');
distancias_BMU1 = distancias_BMU1(:,1);
vecindad_BMU1 = exp(-((distancias_BMU1.^2)/(2*(sigm_loop)^2)));

% Calculando la QEsom1 
store_qe1(estim_idx_sort1(i),:) = min_value1;


%%%%%%%%%%%%%%%%%%%%% SOM2 %%%%%%%%%%%%%%%%%%%%%%

estimulo_loop_2 = estimulo_2(estim_idx_sort2(i),:);
store_estimulo_loop_2(estim_idx_sort2(i),:) = estimulo_loop_2;
%Calculando las diferencias del som
diferencias_2 = pdist2(som_2, estimulo_loop_2, 'euclidean'); 
diferencias_matrix_2 = reshape(diferencias_2, som_high,som_length,1);
%Calculando la activacion del som2
act_som2 = exp(-(diferencias_matrix_2)./max(exp(-(diferencias_matrix_2))));
%Calculando a la ganadora BMU (Best Matching Unit)
min_value2 = min(min((diferencias_matrix_2)));  
[x2, y2] = find(diferencias_matrix_2 == min_value2); 
BMU_2 = [x2, y2];
BMU2_store(estim_idx_sort2(i),:) = BMU_2;
BMU2_coord = BMU2_store; %Almacena las win-coordinates2
BMU2_colum = repmat(BMU_2, neuronas, 1); 
distancias_BMU2 = pdist2(matrix_SOM_2, BMU2_colum,'euclidean'); 
distancias_BMU2 = distancias_BMU2(:,1); 
vecindad_BMU2 = exp(-((distancias_BMU2.^2)/(2*(sigm_loop)^2)));

% Calculando la QEsom2
store_qe2(estim_idx_sort2(i),:) = min_value2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%% Obteniendo vector de estimulación para el som3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
BMUS_vect_estimulation = cat(2, BMU1_store(estim_idx_sort3(i),:), BMU2_store(estim_idx_sort2(i),:));
BMUS_vect_estimulation_store(estim_idx_sort3(i),:) = BMUS_vect_estimulation; 
BMUS_vect_norm = (BMUS_vect_estimulation-0)./(max(length(som_3)- 0)); %Vect de entrada para som_3
BMUS_vect_norm_estimulation_store(estim_idx_sort3(i),:) = BMUS_vect_norm;


%%%%%%%%%%%%%%%%%%%%% SOM3 %%%%%%%%%%%%%%%%%%%%%%

%Calculando las diferencias del som3
diferencias_3 = pdist2(som_3, BMUS_vect_norm, 'euclidean');
diferencias_matrix_3 = reshape(diferencias_3, som_high,som_length,1);
%Calculando la activacion del som3
act_som3 = exp(-(diferencias_matrix_3)./max(exp(-(diferencias_matrix_3))));
%Calculando a la ganadora BMU (Best Matching Unit) y función de vecindad
min_value3 = min(min((diferencias_matrix_3)));  
[x3, y3] = find(diferencias_matrix_3 == min_value3); 
BMU_3 = [x3, y3];
BMU3_store(estim_idx_sort3(i),:) = BMU_3;
BMU3_colum = repmat(BMU_3, neuronas, 1); 
distancias_BMU3 = pdist2(matrix_SOM_3, BMU3_colum,'euclidean'); 
distancias_BMU3 = distancias_BMU3(:,1); 
vecindad_BMU3 = exp(-((distancias_BMU3.^2)/(2*(sigm_loop)^2)));

store_qe3(estim_idx_sort3(i),:) = min_value3;


%%%%%%%%%%%%%%%%%%%%% Conectando tres SOMs mediante conexiones Hebbianas %%%%%%%%%%%%%%%%%%%%%%
HEBBweight_1 = HEBBweight_1 + 1.-(exp(-(betha(t)*act_som1.*act_som3')));
HEBBweight_1 = HEBBweight_1/max(max(HEBBweight_1));
HEBBweight_2 = HEBBweight_2 + 1.-(exp(-(betha(t)*act_som2.*act_som3')));
HEBBweight_2 = HEBBweight_2/max(max(HEBBweight_2));

%%%%% Actualizando los soms con la interaccion de la ecuacion hebbiana
act_indirect_som3 = HEBBweight_1 + HEBBweight_2; % Para el tercer som
act_indirect_som3 = act_indirect_som3/max(max(act_indirect_som3)); %Normalización
som1_act_indirect = act_som2 .* HEBBweight_2; 
som2_act_indirect = act_som1 .* HEBBweight_1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOM1 act bidirec %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Calculando BAU1 (Best Active Unit)y función de vecindad 
min_value_act1 = min(min((som1_act_indirect)));  
[xx1, yy1] = find(som1_act_indirect == min_value_act1); 
BAU_1 = [xx1, yy1];
BAU1_store(estim_idx_sort2(i),:) = BAU_1;%
BAU1_column = repmat(BAU_1, neuronas, 1); 
distancias_BAU1 = pdist2(matrix_SOM_1, BAU1_column,'euclidean'); 
distancias_BAU1 = distancias_BAU1(:,1);
vecindad_BAU1 = exp(-((distancias_BAU1.^2)/(2*(sigm_loop)^2)));

%Activacion final y actualización del som1
activ_final_som1 = (1-lambda)*vecindad_BMU1 + (lambda)*vecindad_BAU1;%(1-L)*actDir+(L)*actIndir
error_1 = estimulo_loop_1 - som_1; 
som_1 = som_1 + (alpha_loop .* activ_final_som1 .* error_1); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOM2 act bidirec %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Calculando BAU2 (Best Active Unit)y función de vecindad %%%%%%%%%%%
min_value_act2 = min(min((som2_act_indirect)));  
[xx2, yy2] = find(som2_act_indirect == min_value_act2); 
BAU_2 = [xx2, yy2];
BAU2_store(estim_idx_sort1(i),:) = BAU_2; 
BAU2_column = repmat(BAU_2, neuronas, 1); 
distancias_BAU2 = pdist2(matrix_SOM_2, BAU2_column,'euclidean'); 
distancias_BAU2 = distancias_BAU2(:,1);
vecindad_BAU2 = exp(-((distancias_BAU2.^2)/(2*(sigm_loop)^2)));

%Activacion final y actualización del som2
activ_final_som2 = (1-lambda)*vecindad_BMU2 + (lambda)*vecindad_BAU2;%(1-L)*actDir+(L)*actIndir
error_2 = estimulo_loop_2 - som_2; 
som_2 = som_2 + (alpha_loop .* activ_final_som2 .* error_2); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOM3 act bidirec %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Calculando BAU3 (Best Active Unit)y función de vecindad 
min_value_act3 = min(min((act_indirect_som3)));  
[xxx1, yyy1] = find(act_indirect_som3 == min_value_act3); 
BAU_3 = [xxx1, yyy1];
BAU3_store(estim_idx_sort3(i),:) = BAU_3;
BAU3_column = repmat(BAU_3, neuronas, 1); 
distancias_BAU3 = pdist2(matrix_SOM_3, BAU3_column,'euclidean'); 
distancias_BAU3 = distancias_BAU3(:,1);
vecindad_BAU3 = exp(-((distancias_BAU3.^2)/(2*(sigm_loop)^2)));

%Activacion final y actualización del som3
activ_final_som3 = (1-lambda)*vecindad_BMU3 + (lambda)*vecindad_BAU3;%(1-L)*actDir+(L)*actIndir
error_3 = BMUS_vect_norm - som_3; 
som_3 = som_3 + (alpha_loop .* activ_final_som3 .* error_3); 


end

store_que1_epocs(:,:,t) = store_qe1;
store_que2_epocs(:,:,t) = store_qe2;
store_que3_epocs(:,:,t) = store_qe3;
a(t,:) = mean(store_que1_epocs(:,:,t));
b(t,:) = mean(store_que2_epocs(:,:,t));
c(t,:) = mean(store_que3_epocs(:,:,t));

 end
 
a_final = a(T,:);
b_final = b(T,:);
c_final = c(T,:);

%%FIN 