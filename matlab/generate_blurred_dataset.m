
function [] = blur_dataset()
    %  BLUR_DATASET generates synthetically blurred images when given pairs of RGB 
    %   and corresponding depth maps. This code is an adaptation of the layered
    %   approach proposed in "A layer-based restoration framework for variable 
    %   aperture photography", Hasinoff, S.W., Kutulakos, K.N., IEEE 11th 
    %   International Conference on Computer Vision, to create a realistic defocus
    %   blur:

    % Authors: Pauline Trouvé-Peloux and Marcela Carvalho.
    % Year: 2017

    % load parameters
    % Uncomment the next line to reproduce experiments from section 3 of the 
    % paper. Change focus on the file to make different tests.
    parameters_blurred_NYUv2;
    % Uncomment the next line to reproduce the pre-training dataset from section
    % 4 of the paper.
    % parameters_DFD_indoor;

    h = waitbar(0,'Initializing waitbar...');
    s = clock;
    
    path_rgb = ['PATH/TO/RGB/IMAGES'];
    path_depth = ['PATH/TO/DEPTH/MAPS'];

    dest_path_rgb = ['DESTINATION/PATH/TO/BLURRED/RGB/IMAGES'];
    dest_path_depth = ['PATH/TO/DEPTH/MAPS'];
    
    create_dir(dest_path_rgb)
    create_dir(dest_path_depth)

    contents_rgb = dir(path_rgb);
    contents_depth = dir(path_depth);

      for i=1:(length(contents_rgb)-2)

        if(rem(i-1,100) == 0)
            s=clock;
        end
        % read images
        im=double(imread([path_rgb contents_rgb(i+2).name]));
<<<<<<< HEAD
        depth=(imread([path_depth contents_depth(i+2).name]));
=======
            depth=(imread([path_depth contents_depth(i+2).name]));
>>>>>>> 8238c86d1f6ee810ff821f9e297ad53743b2d75c

        %conversion into depth values in meters
        depth=double(depth)/(1000.0);

        [im_refoc, ~, ~, D]=refoc_image(im,depth,step_depth,focus,f,N,px,dmode);

<<<<<<< HEAD
        imwrite(uint8(im_refoc), [dest_path_rgb contents_rgb(i+2).name])
        imwrite(uint16(depth*1000), [dest_path_depth contents_depth(i+2).name]) % save depth in milimeters   
=======
        [im_refoc, ~, ~, D]=refoc_image(im,depth,step_depth,focus,f,N,px,mode_);

        imwrite(uint8(im_refoc), [dest_path_rgb contents_rgb(i+2).name])
        imwrite(uint16(depth*1000), [dest_path_depth contents_depth(i+2).name]) % save depth in milimeters   

>>>>>>> 8238c86d1f6ee810ff821f9e297ad53743b2d75c
        if (rem(i-1,100) == 0)
            is = etime(clock, s);
            esttime = is * (length(contents_rgb)-2 -i);
        end

        [hours, min, sec] = sec_hms(esttime - etime(clock,s));

        perc = i/(length(contents_rgb)-2);
        waitbar(perc,h,...
            [' focus: ' num2str(focus) ' ' sprintf('%3.1f%% [%2.0fh%2.0fm%2.0fs]',...
            perc*100, hours, min, sec)]);
      end

%       figure
%       plot(D, sigma_vec)

end

function [] = create_dir(dir_path)
    if(exist(dir_path)~=7)
        mkdir(dir_path)
    else
        display([dir_path  'already exists'])
    end
end
