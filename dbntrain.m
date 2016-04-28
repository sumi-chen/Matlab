function dbn = dbntrain(dbn, x, opts)
    n = numel(dbn.rbm);


    dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, opts);
    for i = 2 : n
        x = rbmup(dbn.rbm{i - 1}, x);  // 即sigm(W*x+c)
        dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x, opts);
    end


end
