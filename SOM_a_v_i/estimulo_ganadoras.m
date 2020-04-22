function [ganadoras_finales, estimulo_loop_final] = estimulo_ganadoras(som, store_estimulo_loop, som_high, som_length)

[ensayos_totales, ~] = size(store_estimulo_loop);

for i=1:ensayos_totales

estimulo_loop = store_estimulo_loop(i,:);
 
diferencias = pdist2(som, estimulo_loop, 'euclidean'); 
diferencias_matrix = reshape(diferencias, som_high,som_length,1);
min_value1 = min(min(abs(diferencias_matrix)));  
[x, y] = find(diferencias_matrix == min_value1);
win = [x, y];
ganadoras_finales(i,:) = win;
estimulo_loop_final(i, :) = estimulo_loop;


end

end