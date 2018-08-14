clc
clear

fpth=imgetfile();
i=imread(fpth);
im=rgb2gray(i);
im=imrotate(im,-90);
im=imcrop(im,[1,1,400,240]);
e = edge(im, 'canny'); 
imshow(e);
radii = 15:1:70;
h = circle_hough(e, radii, 'same', 'normalise');
peaks = circle_houghpeaks(h, radii, 'nhoodxy', 15, 'nhoodr', 21, 'npeaks', 1);

hold on;
for peak = peaks
    [x, y] = circlepoints(peak(3));
    plot(x+peak(1), y+peak(2), 'g-');
end
hold off;


I=i;
I=imrotate(I,-90);
imageSize=size(I);
ci=[peak(2),peak(1),70];
[xx,yy]=ndgrid((1:imageSize(1))-ci(1),(1:imageSize(2))-ci(2));
mask=uint8((xx.^2+yy.^2)<ci(3)^2);
cpi=uint8(zeros(size(I)));
cpi(:,:,1)=I(:,:,1).*mask;
cpi(:,:,2)=I(:,:,2).*mask;
cpi(:,:,3)=I(:,:,3).*mask;
[x,y,z]=size(cpi);
cpi=cpi(1:x/2,:,:);

fd=vision.CascadeObjectDetector();
bbox=step(fd,cpi);
IFaces = insertObjectAnnotation(cpi, 'rectangle', bbox, 'Face');
figure, imshow(IFaces), title('Detection Completed');
[a,b]=size(bbox);
if(a==0)
    msgbox('HELMET detected');
else
    msgbox('FACE without helmet detected');
end