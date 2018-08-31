function [ h, m, s ] = sec_hms(t)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    h = floor(t / 3600);
    t = t - h * 3600;
    m = floor(t / 60);
    s = t - m * 60;
end

