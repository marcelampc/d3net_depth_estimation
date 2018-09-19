% Author: Pauline Trouvé
% Date: 16/03/2017
% Modifications by Marcela Carvalho
% mode = disk, gaussian
% use: generate_blurred_dataset(train, 2, disk)
function [] = generate_blurred_dataset_NYUv2()

    % load parameters
    parameters;

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
            depth=(imread([path_depth contents_depth(i+2).name]));

        %conversion into depth values in meters
        depth=double(depth)/(1000.0);


        [im_refoc, ~, ~, D]=refoc_image(im,depth,step_depth,focus,f,N,px,mode_);

        imwrite(uint8(im_refoc), [dest_path_rgb contents_rgb(i+2).name])
        imwrite(uint16(depth*1000), [dest_path_depth contents_depth(i+2).name]) % save depth in milimeters   

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
