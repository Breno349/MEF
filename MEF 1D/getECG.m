function [elements,coords,groups] = getECG(filename)

    fid = fopen(filename,'r');
    if fid < 0, error('Erro ao abrir o arquivo.'); end
    
    coords = []; elements = []; groups = containers.Map();
    
    while ~feof(fid)
        L = strtrim(fgetl(fid));
        if startsWith(L,'NELEM=')
            nE = sscanf(L,'%*[^=]= %d');
            E = textscan(fid,'%d %d %d %d %d',nE);
            elements = cell2mat(E);
            elements = elements(:,2:4)+1; % tipo, n1, n2, n3 → +1 (índice base 0)
        elseif startsWith(L,'NPOIN=')
            nP = sscanf(L,'%*[^=]= %d');
            P = textscan(fid,'%f %f %*f',nP);
            coords = cell2mat(P);
        elseif startsWith(L,'MARKER_TAG=')
            tag = strtrim(extractAfter(L,'='));
            nL = sscanf(fgetl(fid),'%*[^=]= %d');
            D = textscan(fid,'%d %d %d',nL);
            D = cell2mat(D); D(:,2:3) = D(:,2:3)+1;
            groups(tag) = unique(D(:,2:3));
        end
    end
    fclose(fid);
end