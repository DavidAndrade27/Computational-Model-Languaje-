%%function[Wend] = hebb_red(matriz_A, valor_beta, umbral)


activaciones = input ('matriz de activaciones>> '); 
beta = input ('valor de beta >>'); 
umbral = input('ingresa tu valor de umbral>> ');


[t, neuronas]= size (activaciones);

W =zeros (neuronas);
Woriginal = W;
[xW, yW] = size (W);

Wcoactivacion1 = Woriginal;
Wcoactivacion2 = Woriginal;


for x = 1:t
coactivacion = activaciones (x,:) .* activaciones (x,:)';

Wcoactivacion1 = Wcoactivacion1 + ((coactivacion >= umbral).* (coactivacion-Wcoactivacion1)).*(coactivacion.*beta);
Wcoactivacion2 = Wcoactivacion2 + ((coactivacion <= umbral) .* (coactivacion-Wcoactivacion2)) .* (coactivacion .* beta);

Wcoactivacion1final ((x*xW)-xW+1:x*xW,1:neuronas)= Wcoactivacion1;
Wcoactivacion2final ((x*xW)-xW+1:x*xW,1:neuronas)= Wcoactivacion2;

coac = Wcoactivacion1final - Wcoactivacion2final;

end

Wend = [Woriginal+coac];