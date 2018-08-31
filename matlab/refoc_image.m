function [Im_refoc, Ad, sigma_vector, D]=refoc_image(im, depth, step_depth, focus, f, N, px, type_flou)

% im : gray image or color image
% depth : corresponding depths values
% step_depth :  minimal step of depth values in the depth map
% focus : position of the in-focus plane
% f : focal length of the simulated lens
% N : F number of the simulated lens
% px : pixel size of the simulated lens
% type_flou : gaussian or disk
% !! px, f, focus, depth , step_depth must have the same units

Phi=f/N;
rho=0.3;

% window size for gaussian blur generation
n=21;

D=min(depth(:))-0.5:step_depth:max(depth(:))+0.5;

% initialisation
% im_refoc=zeros(size(im,1),size(im,2));

Ak=zeros(size(im,1),size(im,2),length(D));
% Ap_para=zeros(size(im,1),size(im,2));

sigma_vector = zeros(1, length(D)-1);
% Ad = zeros(1, length(D)-1);
PSF = cell(1, length(D)-1);
Im_refoc = zeros(size(im));

se = strel('disk', 4);
for k=1:length(D)-1
    % PSf simulation
    det=(1/f-1/focus)^(-1);
    if strcmp(type_flou,'gaussian')==1
        sigma=Phi*rho*det*abs(-1/((D(k)+D(k+1))/2)+1/f-1/det)/px;
        PSF{k}=fspecial('gaussian', n, sigma+0.0001);
%         Para(k)=sigma;
        sigma_vector(k) = sigma;
    elseif strcmp(type_flou,'disk')==1
        phi=Phi*det*abs(-1/((D(k)+D(k+1))/2)+1/f-1/det)/px;
        PSF{k}=fspecial('disk',phi/2);
%         Para(k)=phi/2;
    end


    % mask extraction
    temp=zeros(size(im,1),size(im,2));
    temp(depth<D(k+1)&depth>=D(k))=1;
    Ad(:,:,k)=imdilate(temp,se);
     
    
end

% do that to avoid black regions
%Ak(:,:,length(D)-1)=ones(size(im,1),size(im,2));


for m=1:size(im,3)
    
    A1L1= imdilate(Ak(:,:,1),se).*im(:,:,m);
    im_refoc=conv2(A1L1,PSF{1},'same');
    Mk=ones(size(im,1),size(im,2));
    
    for k=2:length(D)-1
      
        AkpLkp2=Ad(:,:,k).*im(:,:,m);
        %AkLk=Ak(:,:,k).*im(:,:,m);
        Mk=Mk.*(1-conv2(Ad(:,:,k-1),PSF{k-1},'same'));
        im_refoc_temp=conv2(AkpLkp2,PSF{k},'same').*Mk;
        im_refoc=im_refoc+im_refoc_temp;
        
    end
    
   Im_refoc(:,:,m)=im_refoc;
 
end
