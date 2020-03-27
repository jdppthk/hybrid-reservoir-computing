function w = generate_reservoir(size, radius, degree)

sparsity = degree/size;

w = sprand(size,size,sparsity);

%[i,j,s] = find(w);

%s = -1 + 2.*s;


%w = sparse(i, j, s);

e = eigs(w);
e = abs(e);
w = (w./max(e))*radius;