function varargout = seg_compare(varargin)
% SEG_COMPARE MATLAB code for seg_compare.fig
%      SEG_COMPARE, by itself, creates a new SEG_COMPARE or raises the existing
%      singleton*.
%
%      H = SEG_COMPARE returns the handle to a new SEG_COMPARE or the handle to
%      the existing singleton*.
%
%      SEG_COMPARE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SEG_COMPARE.M with the given input arguments.
%
%      SEG_COMPARE('Property','Value',...) creates a new SEG_COMPARE or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before seg_compare_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to seg_compare_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help seg_compare

% Last Modified by GUIDE v2.5 12-Jan-2018 08:51:52

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @seg_compare_OpeningFcn, ...
                   'gui_OutputFcn',  @seg_compare_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before seg_compare is made visible.
function seg_compare_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to seg_compare (see VARARGIN)

% Choose default command line output for seg_compare
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes seg_compare wait for user response (see UIRESUME)
% uiwait(handles.figure1);
global num_click
num_click = 0;
global num_picture
num_picture = length(dir('piam_img_256'))-2;
global kmeans_score
kmeans_score = ones([num_picture,1]);
global tair_score
tair_score = ones([num_picture,1]);
global original_name
global kmeans_name
global tair_name
global original_name_cell
global select_name

select_name = {};
original_name={};
kmeans_name={};
tair_name={};
original_name_cell={};
temp = dir('piam_img_256');
tair_temp = dir('piam_img_256_seg');
kmeans_temp = dir('piam_img_extract_256');
for inx=3:length(temp)
    original_name{inx-2} = strcat([temp(inx).folder,'/',temp(inx).name]);
    seg_name_temp = temp(inx).name;
    tair_name{inx-2} =  strcat([tair_temp(inx).folder,'/',seg_name_temp(1:end-4),'_seg.jpg']);
    kmeans_name{inx-2} = strcat([kmeans_temp(inx).folder,'/',temp(inx).name]);
    original_name_cell{inx-2} = temp(inx).name;
end

select_temp = dir('piam_img_256_select_from_kmeans');
for inx=3:length(select_temp)
    select_name{inx-2}=select_temp(inx).name;
end
% --- Outputs from this function are returned to the command line.
function varargout = seg_compare_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global num_click
num_click = num_click + 1;
global num_picture
global kmeans_score
global tair_score
global original_name
global kmeans_name
global tair_name

global original_name_cell
global select_name

if(num_click<=num_picture)
    g = imread(char(original_name{num_click}));
    axes(handles.axes2);
    imshow(g);

    h = imread(char(tair_name{num_click}));
    axes(handles.axes3);
    imshow(h);
    
    f = imread(char(kmeans_name{num_click}));
    axes(handles.axes1);
    imshow(f);
    
    x = get(handles.edit1,'String');
    value = str2double(x);
    kmeans_score(num_click)=value;
    
    y = get(handles.edit2,'String');
    value = str2double(y);
    tair_score(num_click)=value;
    
    if(any(strcmp(select_name,original_name_cell{num_click})))
        text = sprintf('k-means');
        set(handles.text9,'String',text);
    else
        text = sprintf('k-means');
        set(handles.text9,'String',text);        
    end
else
    dlmwrite('kmeans_score.txt',kmeans_score);
    dlmwrite('tair_score.txt',tair_score);
end

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global num_click
num_click = num_click - 1;
if(num_click<1)
    num_click = 1;
end
global num_picture
global kmeans_score
global tair_score
global original_name
global kmeans_name
global tair_name

global original_name_cell
global select_name

if(num_click>0 && num_click<=num_picture)
    g = imread(char(original_name{num_click}));
    axes(handles.axes2);
    imshow(g);

    h = imread(char(tair_name{num_click}));
    axes(handles.axes3);
    imshow(h);
    
    f = imread(char(kmeans_name{num_click}));
    axes(handles.axes1);
    imshow(f);
    
    x = get(handles.edit1,'String');
    value = str2double(x);
    kmeans_score(num_click)=value;
    
    y = get(handles.edit2,'String');
    value = str2double(y);
    tair_score(num_click)=value;
    
    if(any(strcmp(select_name,original_name_cell{num_click})))
        text = sprintf('k-means');
        set(handles.text9,'String',text);
    else
        text = sprintf('tair-net');
        set(handles.text9,'String',text);        
    end
else
    dlmwrite('kmeans_score.txt',kmeans_score);
    dlmwrite('tair_score.txt',tair_score);
end


% --- Executes during object creation, after setting all properties.
function text9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function text10_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
