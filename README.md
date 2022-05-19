// # Calculate number of samples required for an image
clc ;
close ;
m = 6;
n = 4;
N = 100;
N2= N*N ; // Number of dots per inch in both direction
Fs= m* n * N2 ;
disp ( Fs , 'Number of samples requried to preserve the information in the image')

// # spatialresolution
clc;
clear all;
Img1=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\lena.jpeg');
Img = rgb2gray(Img1);
//512*512
subplot (2,2,1),imshow(Img),title('Og image 512*512');
//256*256
Samp=zeros(256);
m=1;
n=1;
for i=1:2:512
    for j=1:2:512
         Samp(m,n)=Img(i,j);
           n=n+1;       
        end 
        n=1;
        m=m+1;  
end
SampImg256=mat2gray(Samp);
subplot(2,2,2);
imshow(SampImg256);
title('Sampled.Img256*256')

// # mean filter
a=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\camera.png');
b1=double(a);
c=imnoise(a,'gaussian');
d=double(c);
b=d;
m=(1/9)*(ones(3,3));
[r1,c1]=size(a);

subplot(2,2,1);
imshow(a);
title('org img');

subplot(2,2,2);
imshow(c);
title('noised img');

for i=2:r1-1
for j=2:c1-1
a1=d(i-1,j-1)+d(i-1,j)+d(i-1,j+1)+d(i,j-1)+d(i,j)+d(i,j+1)
+d(i+1,j-1)+d(i+1,j)+d(i+1,j+1);
b(i,j)=a1*(1/9);
end
end
subplot(2,2,3);
imshow(uint8(b)); 
title('Filtered Image');


///////
Samp=zeros(128);
m=1;
n=1;
for i=1:4:512
    for j=1:4:512
         Samp(m,n)=Img(i,j);
           n=n+1;       
        end 
        n=1;
        m=m+1;  
end
SampImg128=mat2gray(Samp);
subplot(2,2,3),imshow(SampImg128),title('Sampled.Img128*128')

//////////////////////

Samp=zeros(64);
m=1;
n=1;
for i=1:8:512
    for j=1:8:512
         Samp(m,n)=Img(i,j);
           n=n+1;       
        end 
        n=1;
        m=m+1;  
end
SampImg64=mat2gray(Samp);
subplot(2,2,4),imshow(SampImg64),title('Sampled.Img64*64')

// # median filter
a=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\camera.png');
b1 = double(mtlb_double(a));
c = imnoise(a,"salt & pepper",0.2);
d = double(mtlb_double(c));
b = d;
m = (1/9)*ones(3,3);


subplot(2,2,1);
imshow(a);
title('org img');

subplot(2,2,2);
imshow(c);
title('noised img');


[r1,c1] = size(mtlb_double(a));
for i = 2:r1-1
        for j = 2:c1-1
        a1 = [d(i-1,j-1),d(i-1,j),d(i-1,j+1),d(i,j-1),d(i,j), d(i,j+1),d(i+1,j-1),d(i+1,j),d(i+1,j+1)];
        a2 = gsort(a1,"g","i");//gsort(A,'g','i') sort the elements of the array A in the increasing order.
        med = a2(5);
        b(i,j) = med;
        end;
end;
subplot(2,2,3);
imshow(uint8(b)); 
title('Filtered Image');

// # 4-2-Wiener Filter
clc;
clear all;
close;
a=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\camera.png');
x=double(a);
sigma = 50;
gamma = 1;
alpha = 1;
[M N]=size(x);
h = ones(5,5)/25;
Freqa = fft2(x);
Freqh = fft2(h,M,N);
y = real(ifft(Freqh.*Freqa))+25*randn(M,N);

// # basicfunction
clc;
clear all;

i1=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\camera.png');
i2=imread('C:\Users\Mithilesh\Desktop\all class\DIP\dip\New Folder\lena.jpeg');
//Img = rgb2gray(Img);
//Samp=zeros(256);
figure(1);
imshow(i1);
figure(2);
imshow(i2);
figure(3);
subplot(1,2,1);
imshow(i1);
subplot(1,2,2);
imshow(i2);

// # Bit Place slicing
clc;
clear all;
f=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\lenag.jpeg');
f=double(f);
[r,c]=size(f);
com=[128 64 32 16 8 4 2 1];

for k=1:1:length(com);
    for i=1:r
        for j=1:c
        new(i,j)=bitand(f(i,j),com(k));
    end
    subplot(2,4,k);
    imshow(new);
    end
end

// # boundary detection
clc;
clear all;
a=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\rectb.png');
a=rgb2gray(a);
subplot(2,1,1);
imshow(a);
title('org img');
d=a;
[r,c]=size(d);
m=[1 1 1;1 1 1;1 1 1];
for i=2:1:r-1
for j=2:1:c-1
new=[(m(1)*d(i-1,j-1)) (m(2)*d(i-1,j)) (m(3)*d(i-1,j+1))
(m(4)*d(i,j-1)) (m(5)*d(i,j)) (m(6)*d(i,j+1))
(m(7)*d(i+1,j-1)) (m(8)*d(i+1,j)) (m(9)*d(i+1,j+1))];
A2(i,j)=min(new);
aa(i,j)=d(i,j)-A2(i,j);
end
end
subplot(2,1,2);
imshow(aa);title('Boundary Extracted Image');

// # closing
clc;
clear all;
a=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\rice.jpeg');
a=rgb2gray(a);
d=a;
A2=d;
A1=d;
subplot(2,2,1);
imshow(a);
title('org img');
[r,c]=size(d);
m=[1 1 1;1 1 1;1 1 1];
for i=2:1:r-1
for j=2:1:c-1
new=[(m(1)*d(i-1,j-1)) (m(2)*d(i-1,j)) (m(3)*d(i-1,j+1))
(m(4)*d(i,j-1)) (m(5)*d(i,j)) (m(6)*d(i,j+1))
(m(7)*d(i+1,j-1)) (m(8)*d(i+1,j)) (m(9)*d(i+1,j+1))];
A2(i,j)=max(new);
end
subplot(2,2,2);
imshow(A2);
title('org img');
end

d = A2;
A1=A2;
[r,c]=size(d);
for i=2:1:r-1
for j=2:1:c-1
new=[(m(1)*d(i-1,j-1)) (m(2)*d(i-1,j)) (m(3)*d(i-1,j+1))
(m(4)*d(i,j-1)) (m(5)*d(i,j)) (m(6)*d(i,j+1))
(m(7)*d(i+1,j-1)) (m(8)*d(i+1,j)) (m(9)*d(i+1,j+1))];
A1(i,j)=min(new);
end
subplot(2,2,3);
imshow(A1);title('Processed Image - Closing');
end

// # Pract no2: Piecewise linear transformations
//a. Contrast Stretching
clc;
clear all;
a=imread('C:\Users\Desktop\all class\DIP\All ocdes\image\lena.jpeg');
a=rgb2gray(a);
subplot(2,1,1);
imshow(a);
title('org img');
T=60; //threshold value
[r,c]=size(a);
for i=1:r
    for j=1:c
        if (a(i,j)<=T)
            x(i,j)=0;
        else
            x(i,j)=255;
        end
        x=uint8(x);
subplot(2,1,2);
imshow(x);
title('threshholded img');
    end
end

// # d-e add sub
clc;
clear all;
A=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\camera.png');
B=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\circle.png');
A=rgb2gray(A);
B=rgb2gray(B);
C=imadd(B, A);

D=imsubtract(B, A);
figure(1);
subplot(2,2,1);
imshow(A);
subplot(2,2,2);
imshow(B);
subplot(2,2,3);
imshow(C);
subplot(2,2,4);
imshow(D);

// # dialation
clc;
clear all;
a=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\rectb.png');
a=rgb2gray(a);
d=a;
A1=a;
[r,c]=size(d);
subplot(2,1,1);
imshow(a);
title('org img');
m=[1 1 1;1 1 1;1 1 1];
// m=ones(5,5);
for i=2:1:r-1
for j=2:1:c-1
new=[(m(1)*d(i-1,j-1)) (m(2)*d(i-1,j)) (m(3)*d(i-1,j+1)) (m(4)*d(i,j-1)) (m(5)*d(i,j)) (m(6)*d(i,j+1)) (m(7)*d(i+1,j-1)) (m(8)*d(i+1,j)) (m(9)*d(i+1,j+1))];
A1(i,j)=max(new);
end
subplot(2,1,2);
imshow(A1);title('Processed Image - dilation');
end

// # erosion
clc;
clear all;
a=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\rectb.png');
a=rgb2gray(a);
subplot(2,1,1);
imshow(a);
title('org img');
A1=a;
d=a;
[r,c]=size(d);
m=[1 1 1;1 1 1;1 1 1];
// m=ones(5,5);
for i=2:1:r-1
for j=2:1:c-1
    new=[(m(1)*d(i-1,j-1)) (m(2)*d(i-1,j)) (m(3)*d(i-1,j+1)) (m(4)*d(i,j-1)) (m(5)*d(i,j)) (m(6)*d(i,j+1)) (m(7)*d(i+1,j-1)) (m(8)*d(i+1,j)) (m(9)*d(i+1,j+1))];
A1(i,j)=min(new);
end
subplot(2,1,2);
title('org img');imshow(A1);title('Processed Image - Erosion');
end

// # grey level with bg
clc;
clear all;
a=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\camera.png');

a1=58; // This value is user defined
b1=158; // This value is user defined
[r,c]=size(a);
figure(2);
subplot(2,1,1);
imshow(a);
for i=1:r
    for j=1:c
        if (a(i,j)>a1 & a(i,j)<b1)
            x(i,j)=255;
        else
            x(i,j)=a(i,j);
        end
    end
end
x=uint8(x);
subplot(2,1,2);
imshow(x);

// # grey level without bg
clc;
clear all;
a=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\camera.png');

a1=50; // This value is user defined
b1=150; // This value is user defined
[r,c]=size(a);
figure(1)
subplot(2,1,1);
imshow(a);
for i=1:r
    for j=1:c
        if (a(i,j)>a1 & a(i,j)<b1)
            x(i,j)=255;
        else
            x(i,j)=0;
        end
    end
end
x=uint8(x);
subplot(2,1,2);
imshow(x);

// # grey to color
clc;
close;

a = imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\lenag.jpeg');

//Displaying Original RGB image
figure(1);
imshow(a);
title("Original Image")

//Displaying Gray level image
b = rgb2gray(a);
figure(2);
imshow(b);
title("Gray Level Image")

//Displaying False coloring(Pseudo) image
figure(3)
imshow(b,jetcolormap(256));
title("Pseudo Color Image");

// # grey to false color
clc;
close;
a = imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\lenag.jpeg');

//Displaying Original RGB image
figure(1);
imshow(a);
title("Original Image")

//Displaying Gray level image
b = rgb2gray(a);
figure(2);
imshow(b);
title("Gray Level Image")

//Displaying False coloring(Pseudo) image
figure(3)
imshow(b,jetcolormap(256));
title("Pseudo Color Image");

// # highpass
a=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\camera.png');
b=double(a);
c=imnoise(a,'salt & pepper',0.2);
d=double(c);

subplot(2,2,1);
imshow(a);
title('original');
subplot(2,2,2);
imshow(c);
title('noise img');
m=[-1 -1 -1;-1 8 -1;-1 -1 -1];
[r1,c1]=size(a);

for i=2:1:r1-1
    for j=2:1:c1-1
   new(i,j)=(m(1)*d(i-1,j-1))+(m(2)*d(i-1,j))+(m(3)*d(i-1,j+1))    +(m(4)*d(i,j-1))+(m(5)*d(i,j))+(m(6)*d(i,j+1))
    +(m(7)*d(i+1,j-1))+(m(8)*d(i+1,j))+(m(9)*d(i+1,j+1));
    end
end
subplot(2,2,3);
imshow(uint8(new));
title('LP-filtered img');

// # historgram
//Q 2_B 1. Program to apply histogram equalization
clc;
clear all;
a=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\lena.jpeg');
a=rgb2gray(a);
h=zeros(1,258);
[r,c]=size(a);
for i=1:r
    for j=1:c
        if (a(i,j)==0)
            h(0)=h(0)+1;
        end
        k=a(i,j);
        h(k)=h(k)+1;
    end
end
figure(1);
subplot(1,2,1);
imshow(uint8(a));
title('Original Image')
subplot(1,2,2);
bar(h);
title('Image histogram');

// # histogram color image
//Q 2_B 1. Program to plot the histogram of an image and categorise
clc;
clear all;
image=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\lena.png');


R = image(:,:,1);
G = image(:,:,2);
B = image(:,:,3);
nBins = 256;
//Get histValues for each channel
[yR,x] = imhist(R,nBins);
[yG,x] = imhist(G,nBins);
[yB,x] = imhist(B,nBins);
//Plot them together in one plot
plot(x,yR,x,yG,x,yB,"Linewidth",2);
xlabel("RGB Intensity");
ylabel("No. of Pixels");
set(gca(),"grid",[1,1]);

// # image negative
//for gray image
clc;
clear all;
A = imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\camera.png');
A=rgb2gray(A);
subplot(2,1,1);
imshow(A);
title('Original Image ');

[row col]=size(A);
for x=1:row
    for y=1:col
       A(x,y)=255-A(x,y);
    end
end
subplot(2,1,2);
imshow(A);
title('Image after negation');

// # imagentation color
clc;
clear all;
A = imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\negimg.jpg');

subplot(2,1,1);
imshow(A);
title('Orignial Image');
R = A(:,:,1); 
G = A(:,:,2); 
B = A(:,:,3);

[row col]=size(A);
for x=1:row
    for y=1:col
       R(x,y)=255-R(x,y);
       G(x,y)=255-G(x,y);
       B(x,y)=255-B(x,y);
    end
end

A(:,:,1)=R; 
A(:,:,2)=G; 
A(:,:,3)=B;


subplot(2,1,2);
imshow(A);
title('Image after negation');

// # intensity level
clc;
clear all;
figure(1)

subplot(3,3,1);
i=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\lena.jpeg');
imshow(i);
title('original image');
subplot(3,3,2);
j1=imresize(i,0.8);
imshow(j1);
title('resized image 0.8');

subplot(3,3,3);
j2=imresize(i,0.7);
imshow(j2);
title('resized image 0.7');

subplot(3,3,4);
j3=imresize(i,0.6);
imshow(j3);
title('resized image 0.6');

subplot(3,3,5);
j4=imresize(i,0.1);
imshow(j4);
title('resized image 0.1');

// # log trans
//Pract no 2_A_iii :Program to perform Log transformation
clc;
clear all;
a=imread('C:\Desktop\all class\DIP\All ocdes\image\camera.png');
a=rgb2gray(a);
subplot(2,1,1);
imshow(a);
c=1;
[r1,c1]=size(a);
for i=1:r1
    for j=1:c1
        b=double(a(i,j));
        s(i,j)=c*log10(1+b);
    end
end
new1=uint8(s*100);
//imshow(new1);
subplot(2,2,2);
imshow(new1);

// # lowpass
clc;
clear all;
a=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\camera.png');
b=double(a);
c=imnoise(a,'salt & pepper',0.2);
d=double(c);

subplot(2,2,1);
imshow(a);
title('original');
subplot(2,2,2);
imshow(c);
title('noise img');
m=(1/9)*(ones(3,3));
[r1,c1]=size(a);

for i=2:1:r1-1
    for j=2:1:c1-1
   new(i,j)=(m(1)*d(i-1,j-1))+(m(2)*d(i-1,j))+(m(3)*d(i-1,j+1))    +(m(4)*d(i,j-1))+(m(5)*d(i,j))+(m(6)*d(i,j+1))
    +(m(7)*d(i+1,j-1))+(m(8)*d(i+1,j))+(m(9)*d(i+1,j+1));
    end
end
subplot(2,2,3);
imshow(uint8(new));
title('LP-filtered img');

// # one color to other
clc;
close;
x=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\lena.png');

//Displayimg RGB image
figure(1);
imshow(x);
title('Original RGB Image')

//Displaying HSV image
y = rgb2hsv(x);
figure(2);
imshow(y);
title('HSV version of RGB original Image')

// # opening ex
clc;
clear all;
a=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\rice.jpeg');
a=rgb2gray(a);
d=a;
A2=d;
A1=d;
subplot(2,2,1);
imshow(a);
title('org img');
[r,c]=size(d);
m=[1 1 1;1 1 1;1 1 1];
for i=2:1:r-1
for j=2:1:c-1
new=[(m(1)*d(i-1,j-1)) (m(2)*d(i-1,j)) (m(3)*d(i-1,j+1))
(m(4)*d(i,j-1)) (m(5)*d(i,j)) (m(6)*d(i,j+1))
(m(7)*d(i+1,j-1)) (m(8)*d(i+1,j)) (m(9)*d(i+1,j+1))];
A2(i,j)=min(new);
end
subplot(2,2,2);
imshow(A2);
title('org img');
end

d = A2;
A1=A2;
[r,c]=size(d);
for i=2:1:r-1
for j=2:1:c-1
new=[(m(1)*d(i-1,j-1)) (m(2)*d(i-1,j)) (m(3)*d(i-1,j+1))
(m(4)*d(i,j-1)) (m(5)*d(i,j)) (m(6)*d(i,j+1))
(m(7)*d(i+1,j-1)) (m(8)*d(i+1,j)) (m(9)*d(i+1,j+1))];
A1(i,j)=max(new);
end
subplot(2,2,3);
imshow(A1);title('Processed Image - Opening');
end

// # power law
clc;
clear all;
a=imread('C:\Users\Mithilesh\Desktop\all class\DIP\All ocdes\image\camera.png');
[r,c]=size(a);
subplot(2,1,1);
imshow(a);
G=0.8;
for i=1:r
    for j=1:c
        b=double(a(i,j));
        x(i,j)=b^G;
        end
end
new1=uint8(x);
subplot(2,1,2);
imshow(new1);
