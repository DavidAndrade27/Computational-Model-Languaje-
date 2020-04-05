clearvars; 

estimulo = csvread("estimulos.csv"); 
[ensayos_totales, atributos] = size(estimulo);

neuronas = 100;
som = rand(neuronas, atributos); %som lo compone las neuronas y los 
%atributos que cada una de ellas puede procesar. Los atributos dependen de
%las propiedades de los est�mulos, representadas por las columnas y los
%estimulos estar�an representados por las filas

cellSOM = cell(10, 10); %Preguntar si generar la matriz de 10x10 es un estandar

alpha = 0.7; %El alfa en un principio se mantiene est�tico 
s = 6;

l = 0; 

    while l < 10
    l = l + 1;
    h = 0;
    while h < 10
        h = h + 1;
        cellSOM(h,l) = {[h,l]};
    end
    end
    
cellSOM_column = reshape(cellSOM, 100, 1); %Genera un arreglo columnar con 
%Con las coordenadas de cada posici�n de la matriz. 
matrix_SOM = cell2mat(cellSOM_column); %Nos regresa una matriz 100x2 del 
%arreglo columnar lo que nos da las coordenadas es sus dos posiciones de
%manera separada 
 T = 50; 
for t=1:50 %Establecer �pocas
    
for i=1:ensayos_totales %Este es el loop para hacer el c�lculo con cada uno
    %De los est�mulos que se le presente. 
    
estimulo_loop = estimulo(i,:); %Esta nueva variable va tomando los valores 
%de cada est�mulo hasta cubrirlos todos (las filas)

diferencias = pdist2(som, estimulo_loop, 'euclidean'); %Resta los valores de
%los atributos del estimulo a cada una de las neuronas (las 100 neuronas
%del som)
diferencias_matrix = reshape(diferencias, 10,10,1); 
min_value1 = min(min(abs(diferencias_matrix))); %Determina el valor m�s peque�o
[x, y] = find(diferencias_matrix == min_value1); %Busca el valor m�s peque�o
win = [x, y];

win_2 = repmat(win, 100, 1);  %Repite las coordenadas de la ganadora para despu�s
%sacara la distancia euclidiana que tiene la ganadora vs cada neurona
distancias = pdist2(matrix_SOM, win_2,'euclidean'); 

distancias = distancias(:,1); %Toma a la mejor 
distancias_matrix = reshape(distancias, 10,10); 
distancias_norm = 1 - (distancias/max(distancias)); %Esta ser�a N /Funci�n vecindad

error = estimulo_loop - som; 

som = som + (alpha .* distancias_norm .* error); %Funcion de actualizaci�n
end

%aqu� vamos integrando el descneso de sigma con s = s(1-t/T). Tenemos una
%restricci�n con un if quiza si s > 1 entonces que usea el resultado 
%else emtonces que use s=1 

%Lo mismo podemos hacer para alfa a = a(1-t/T)

end
%%FIN 