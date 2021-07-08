
%imgArrayUpperL=imageDatastore('D:\Tesi\DataInc\train','IncludeSubFolders',true','LabelSource','foldernames');
%imgArrayUpperL=imageDatastore('D:\Tesi\DataInc\validate','IncludeSubFolders',true','LabelSource','foldernames');
imgArrayUpperL=imageDatastore('D:\Tesi\DataInc\test','IncludeSubFolders',true','LabelSource','foldernames');
if( ~exist('D:\Tesi\DataIncRGB\test', 'dir') )
    mkdir('D:\Tesi\DataIncRGB\test');
    if( ~exist('D:\Tesi\DataIncRGB\test/normal', 'dir') )
        mkdir('D:\Tesi\DataIncRGB\test/normal');
    end
    if( ~exist('D:\Tesi\DataIncRGB\test/pneumonia', 'dir') )
        mkdir('D:\Tesi\DataIncRGB\test/pneumonia');
    end
    if( ~exist('D:\Tesi\DataIncRGB\test/covid-19', 'dir') )
        mkdir('D:\Tesi\DataIncRGB\test/covid-19');
    end
end
index=0;
word="test";
while hasdata(imgArrayUpperL)
    [grayImage,info]=read(imgArrayUpperL);
    filler = zeros(size(grayImage),'uint8');
    rgbImage = cat(3, grayImage, grayImage, grayImage);

    name=strcat(word,string(index),".png");
    if(info.Label=="covid-19 ")
        imwrite(rgbImage,strcat("D:\Tesi\DataIncRGB\test/covid-19/",name));
    end
    if(info.Label=="normal")
         imwrite(rgbImage,strcat("D:\Tesi\DataIncRGB\test/normal/",name));
    end
    if(info.Label=="pneumonia")
         imwrite(rgbImage,strcat("D:\Tesi\DataIncRGB\test/pneumonia/",name));
    end
    index=index+1;
end