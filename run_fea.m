%% Finite Element Analysis of 3D truss and frame
% Author: Qiren Gao, Michigan State University
% Latest modifications made by Abhiroop Ghosh, Michigan State University
function [weight, compliance, stress, strain, U, x0_new] = run_fea(Coordinates, Connectivity, fixednodes, loadn, force, density, elastic_modulus)
    % Coordinates -> mm
    % Radius -> Connectivity(:, 3) -> mm
    % Density -> kg/mm3
    
%    strtype = 'frame';
     strtype = 'truss';

    preXr = Coordinates;
    Xr = preXr';
    preconnec0 = Connectivity(:,1:2);
    connec0 = preconnec0';
    numel0 = length(connec0);
    nx = length(Xr);

    x0=Xr(1:3,:);
    R = Connectivity(:,3)';

    RadiusNEL=zeros(numel0,1);
    lengths=zeros(numel0,1);
    for nel=1:numel0
        nodes=connec0(:,nel)';
        RadiusNEL(nel,1) = R(1,nel);
        xnel = x0(:,nodes);
        dx=xnel(1,2)-xnel(1,1);
        dy=xnel(2,2)-xnel(2,1);
        dz=xnel(3,2)-xnel(3,1);
        lnel=sqrt(dx*dx+dy*dy+dz*dz);
        lengths(nel,1)=lnel;
    end
%     nel = 1:numel0;
%     nodes=connec0(:,nel)';
%     RadiusNEL(nel,1) = R(1,nel);
%     xnel = x0(:,nodes');
%     dx=xnel(1,2*nel)-xnel(1,2*nel-1);
%     dy=xnel(2,2*nel)-xnel(2,2*nel-1);
%     dz=xnel(3,2*nel)-xnel(3,2*nel-1);
%     lnel=sqrt(dx.*dx+dy.*dy+dz.*dz);
%     lengths(nel,1)=lnel;

    Fx0 = force(1);
    Fy0 = force(2);
    Fz0 = force(3);

    numel=numel0;
    connec=connec0;
    x=x0;
    ndf=6;
    numnodes = size(x,2);
    neq=numnodes*ndf;
    tdofs=[1:6:neq;2:6:neq;3:6:neq]';
    tdofs=reshape(tdofs',3*numnodes,1);

    %==========================================================================
    % Boundary Conditions: Fixed nodes and dof
    % Type of b.c.:
    %      1,1,1 means "x,y,z disp fixed"
    %      0,1,1 means "y,z   disp fixed"
    %      1,0,0 means "x     disp fixed"   etc...
    %==========================================================================

%     xmin= min(x');
%     xmax= max(x');
    xmin= min(x, 1);
    xmax= max(x, 1);
    eps=norm(xmax-xmin)*1e-5;
    %     fixednodes=find(abs(x(1,:)-min(x(1,:)))<eps);
    fixednodes = fixednodes';
    fixeddof=repmat([1,1,1,1,1,1],size(fixednodes,2),1);


    %==========================================================================
    % f truss, remove rotations
    %==========================================================================
    if strcmp(strtype ,'truss')
        tmp=repmat([0,0,0,1,1,1],numnodes,1);
        fixeddof=[fixeddof;tmp];
        fixednodes=[fixednodes,1:numnodes];
    end

    [freedofs,fixeq] = setbc(numnodes,ndf,fixednodes,fixeddof);


    loadn = loadn';

    nFx = 6*(loadn-1)+1;
    nFy = 6*(loadn-1)+2;
    nFz = 6*(loadn-1)+3;

    % elastimod   = 2850*ones(1,numel);  %   MPa  N/mm^2
    elastimod = elastic_modulus * ones(1,numel);  %   MPa  N/mm^2
    shearmod = elastimod /(2*(1+ 0.33));

    [skBy,skBz,skA,skT] = ...
        buildKBeamLatticeUnscaled(numel,x,connec,elastimod,shearmod);
    if strcmp(strtype ,'truss')
        skBy=zeros(size(skBy));
        skBz=zeros(size(skBz));
        skT=zeros(size(skT));
%         skBy=zeros(144, numel);
%         skBz=zeros(144, numel);
%         skT=zeros(144, numel);
%     else
%         [skBy,skBz,skA,skT] = ...
%         buildKBeamLatticeUnscaled(numel,x,connec,elastimod,shearmod);
    end

    %==========================================================================
    %  precompute   loads
    %==========================================================================
    F=sparse(neq,1);
    F(nFx,1)=Fx0/length(loadn);
    %     F(nFx2,1)=Fx2/length(loadn2);
    F(nFy,1)=Fy0/length(loadn);
    F(nFz,1)=Fz0/length(loadn);

    xval =  2*RadiusNEL;

    areas=pi*xval.*xval/4 ;
    bendingIZ = pi*xval.^4/64.;
    bendingIY = pi*xval.^4/64.;
    torsionJ = bendingIY + bendingIZ;
    %=========================== FEM analysis only ============================
    [sk] = assembleKBeamLattice(numel,neq,connec,...
        areas,bendingIZ,bendingIY,torsionJ,skBy,skBz,skA,skT);
    
    % Deformation of nodes for every dof
    U=zeros(neq,1);
    U(freedofs,:)=sk(freedofs,freedofs)\F(freedofs,:);
    if any(isnan(U))
        compliance = 10000;
    else
        compliance=0;
        compliance = compliance +U(:,1)'*F(:,1);
    end
    final_volume = sum(lengths.*areas);

%     csvwrite(final_compliance_file_name,compliance);
%     csvwrite(final_volume_file_name,final_volume);
    weight = final_volume * density;
    
    x0_new = zeros(size(x0));
    % Get the new coordinates of each node based on the displacements U
    for ii = 1:nx
        x0_new(1:3, ii) = x0(1:3, ii) + U(6 * (ii-1) + 1: 6 * (ii-1) + 3);
    end
%     x0_new(1:3, 1:nx) = x0(1:3, 1:nx) + U(6 * (1:nx-1) + 1: 6 * (1:nx-1) + 3);
    
    new_lengths = zeros([numel, 1]);
    strain = zeros([numel, 1]);
    stress = zeros([numel, 1]);
    % Get the new lengths of members from the connectivity matrix connec0
    for ii = 1:numel0
        elem_nodes = connec0(:, ii)';
        new_lengths(ii) = norm(x0_new(:, elem_nodes(1)) - x0_new(:, elem_nodes(2)));
        strain(ii) = (new_lengths(ii) - lengths(ii)) / lengths(ii);
        stress(ii) = elastic_modulus * strain(ii);
    end
%     elem_nodes = connec0(:, 1:numel0)';
%     new_lengths(1:numel0) = norm(x0_new(:, elem_nodes(1)) - x0_new(:, elem_nodes(2)));
%     strain(1:numel0) = (new_lengths(1:numel0) - lengths(1:numel0)) ./ lengths(1:numel0);
%     stress(1:numel0) = elastic_modulus * strain(1:numel0);
    
    x0_new = x0_new';
    
    %==========================================================================
    function [freedofs,fixeq] = setbc(numnodes,ndf,fixednodes,fixeddof)
    %==========================================================================

    numfix = size(fixednodes,2);

    % Total number of equations
    neq=numnodes*ndf;

    % Apply  fixed boundary condition.  Build list of fixed dof.
    ndoffixed=sum(sum(fixeddof(1:numfix,1:ndf)));
    fixeq=zeros(1,ndoffixed);
    j=0;
    for i=1:numfix
        nodei=fixednodes(i);
        for idof = 1:ndf
            id=fixeddof(i,idof);
            if(id==1)
                j=j+1;
                gieq=(nodei-1)*ndf+idof;
                fixeq(j)=gieq;
            end
        end
    end
    alldofs     = 1:neq;
    freedofs    = setdiff(alldofs,fixeq);
    end

    %==========================================================================
    function [df0dx,dfdx] = gradientsKBeamLattice(numel,numdesvar,mcons,xval,...
        connec,U,lengths,skBy,skBz,skA,skT)
    %==========================================================================
    dfdx=zeros(mcons,numdesvar);
    df0dx =zeros(numdesvar,1);

    for nel=1:numel
        nodes=connec(:,nel)';
        ndofs = reshape([6*(nodes-1)+1; 6*(nodes-1)+2; 6*(nodes-1)+3;...
            6*(nodes-1)+4; 6*(nodes-1)+5;  6*nodes],12,1)';
        ue=U(ndofs,:);
        xv=xval(nel);
        dkby=reshape(skBy(:,nel),12,12);
        dkbz=reshape(skBz(:,nel),12,12);
        dka=reshape(skA(:,nel),12,12);
        dkt=reshape(skT(:,nel),12,12);
        dk = pi*( (4*xv^3/64)*( dkby + dkbz + 2*dkt ) + 2*xv/4*dka );
        df0dx(nel,1)= - ue(:,1)'*( dk )* ue(:,1);
        dfdx(1,nel)= pi*lengths(nel)*xv/2;
    end   % end of gradients
    end

    %==========================================================================
    function [sk] = assembleKBeamLattice(numel,neq,connec,...
        areas,bendingIZ,bendingIY,torsionJ,skBy,skBz,skA,skT)
    %==========================================================================
    sk=sparse(neq,neq);
    for nel=1:numel
        %     anel=areas(nel);
        %     jnel=torsionJ(nel);
        %     inelZ=bendingIZ(nel);
        %     inelY=bendingIY(nel);
        nodes=connec(:,nel)';

        ndofs = reshape([6*(nodes-1)+1; 6*(nodes-1)+2; 6*(nodes-1)+3;...
            6*(nodes-1)+4; 6*(nodes-1)+5;  6*nodes],12,1)';

        sk(ndofs,ndofs)=sk(ndofs,ndofs)+ ...
            reshape(bendingIY(nel)*skBy(:,nel),12,12)+...
            reshape(bendingIZ(nel)*skBz(:,nel),12,12)+...
            reshape(areas(nel)*skA(:,nel),12,12)+...
            reshape(torsionJ(nel)*skT(:,nel),12,12);
    end   % end of assembly
    end

    %==========================================================================
    function [skBy,skBz,skA,skT] = buildKBeamLatticeUnscaled(numel,x,connec,elastimod,shearmod)
    %==========================================================================

    skBy=sparse(144,numel);
    skBz=sparse(144,numel);
    skA=sparse(144,numel);
    skT=sparse(144,numel);
    % rotArray=sparse(144,numel);

    for nel=1:numel

        % Information about element "nel"
        nodes=connec(:,nel)';
        xnel = x(:,nodes);

        enel=elastimod(nel);
        gnel=shearmod(nel);
        [kbbz,kbby,kaxial,ktorsion,rot12]=beam3d(xnel,1,1,1,1,enel,gnel,0);

        skBy(:,nel)=reshape(rot12'*kbby*rot12,144,1);
        skBz(:,nel)=reshape(rot12'*kbbz*rot12,144,1);
        skA(:,nel)=reshape(rot12'*kaxial*rot12,144,1);
        skT(:,nel)=reshape(rot12'*ktorsion*rot12,144,1);
        %     rotArray(:,nel)=reshape(rot12,144,1);


    end   % end of assembly
    end


    %==========================================================================
    function [kbbz12,kbby12,kaxial12,ktorsion12,rot12]=...
        beam3d(xnel,anel,jnel,inelZ,inelY,enel,gnel,tnel)
    %==========================================================================
    dofz = [2, 6, 8, 12];
    dofy = [3, 5, 9, 11];
    dofa = [1, 7];
    dofj = [4, 10];
    dx=xnel(1,2)-xnel(1,1);
    dy=xnel(2,2)-xnel(2,1);
    dz=xnel(3,2)-xnel(3,1);
    lnel=sqrt(dx*dx+dy*dy+dz*dz);
    l=dx/lnel;
    m=dy/lnel;
    n=dz/lnel;

    D=sqrt(l^2+m^2);
    if D > 0
        rot33 = [ l, m, n;  -m/D, l/D, 0; -l*n/D, -m*n/D, D];
    else
        rot33 = [ 0, 0, sign(n);  -1, 0, 0;  0, -sign(n), 0];
    end

    rot12=sparse(12,12);
    rot12([1,2,3],[1,2,3])=rot33;
    rot12([1,2,3]+3,[1,2,3]+3)=rot33;
    rot12([1,2,3]+6,[1,2,3]+6)=rot33;
    rot12([1,2,3]+9,[1,2,3]+9)=rot33;


    kz=enel*inelZ/lnel^3;
    ky=enel*inelY/lnel^3;
    kg=(tnel/lnel/30);
    kbbZ = [12,       6*lnel,    -12,      6*lnel;  ...
        6*lnel,   4*lnel^2,  -6*lnel,  2*lnel^2;  ...
        -12,      -6*lnel,     12,     -6*lnel;  ...
        6*lnel,   2*lnel^2,  -6*lnel,  4*lnel^2];

    kbbY = [ 12,       -6*lnel,    -12,      -6*lnel;  ...
        -6*lnel,    4*lnel^2,   6*lnel,  2*lnel^2;  ...
        -12,        6*lnel,     12,      6*lnel;  ...
        -6*lnel,     2*lnel^2,   6*lnel,  4*lnel^2];

    kbbz= kz*kbbZ;
    kbby= ky*kbbY;
    % kggZ=kg*[    36,    3*lnel,    -36,       3*lnel; ...
    %     3*lnel,    4*lnel^2, -3*lnel,   -lnel^2; ...
    %     -36,   -3*lnel,     36,      -3*lnel; ...
    %     3*lnel,   -lnel^2,   -3*lnel,   4*lnel^2];
    %
    % kggY=kg*[   36,    -3*lnel,    -36,      -3*lnel; ...
    %     -3*lnel,    4*lnel^2,  3*lnel,   -lnel^2; ...
    %     -36,     3*lnel,     36,       3*lnel; ...
    %     -3*lnel,   -lnel^2,    3*lnel,   4*lnel^2];
    % kgg=kg*[36, 3*lnel, -36, 3*lnel; 3*lnel, 4*lnel^2, -3*lnel, -lnel^2; -36, -3*lnel, 36, -3*lnel; 3*lnel, -lnel^2, -3*lnel, 4*lnel^2];

    kaxial=(enel*anel/lnel)*[1,-1;-1,1];
    ktorsion=(gnel*jnel/lnel)*[1,-1;-1,1];

    kbbz12 = sparse(12,12); kbbz12(dofz, dofz) =  kbbz   ;
    kbby12 = sparse(12,12); kbby12(dofy, dofy) =  kbby   ;
    kaxial12 =   sparse(12,12); kaxial12(dofa, dofa) =   kaxial;
    ktorsion12 =   sparse(12,12); ktorsion12(dofj, dofj) =   ktorsion;
    end
end
