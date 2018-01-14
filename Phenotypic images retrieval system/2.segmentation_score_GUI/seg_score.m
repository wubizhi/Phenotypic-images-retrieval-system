function varargout = seg_score(varargin)
% SEG_SCORE MATLAB code for seg_score.fig
%      SEG_SCORE, by itself, creates a new SEG_SCORE or raises the existing
%      singleton*.
%
%      H = SEG_SCORE returns the handle to a new SEG_SCORE or the handle to
%      the existing singleton*.
%
%      SEG_SCORE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SEG_SCORE.M with the given input arguments.
%
%      SEG_SCORE('Property','Value',...) creates a new SEG_SCORE or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before seg_score_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to seg_score_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help seg_score

% Last Modified by GUIDE v2.5 12-Jan-2018 19:50:34

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @seg_score_OpeningFcn, ...
                   'gui_OutputFcn',  @seg_score_OutputFcn, ...
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


% --- Executes just before seg_score is made visible.
function seg_score_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to seg_score (see VARARGIN)

% Choose default command line output for seg_score
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes seg_score wait for user response (see UIRESUME)
% uiwait(handles.figure1);
global num_click
num_click = 0;
global num_picture
num_picture = length(dir('piam_img_256'))-2;
global score_array
score_array = ones([num_picture,1]);
global original_name
global segmentation_name
original_name={};
segmentation_name={};
temp = dir('piam_img_256');
% seg_temp = dir('piam_img_256_seg');
seg_temp = dir('piam_img_extract_256');
for inx=3:length(temp)
    original_name{inx-2} = strcat([temp(inx).folder,'/',temp(inx).name]);
    seg_name_temp = temp(inx).name;
    segmentation_name{inx-2} =  strcat([seg_temp(inx).folder,'/',seg_name_temp]);
%     original_name = [original_name;strcat([temp(inx).folder,'/',temp(inx).name])];
end


% --- Outputs from this function are returned to the command line.
function varargout = seg_score_OutputFcn(hObject, eventdata, handles) 
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


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global num_click
num_click = num_click + 1
global num_picture
global score_array
global original_name
global segmentation_name

if(num_click<=num_picture)
%     g_path = strcat(['./img_original/t',num2str(num_click),'.jpg']);
    g = imread(char(original_name{num_click}));
    axes(handles.axes1);
    imshow(g);

%     h_path = strcat(['./img_segmentation/t',num2str(num_click),'_seg.jpg']);
    h = imread(char(segmentation_name{num_click}));
    axes(handles.axes2);
    imshow(h);
    
    x = get(handles.edit1,'String');
    value = str2double(x);
    score_array(num_click)=value;
else
    dlmwrite('score.txt',score_array);
end
% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over pushbutton1.
function pushbutton1_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object creation, after setting all properties.
function axes1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes1


% --- Executes during object creation, after setting all properties.
function axes2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes2


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over edit1.
function edit1_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


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
global score_array
global original_name
global segmentation_name

if(num_click>0 && num_click<num_picture)
%     g_path = strcat(['./img_original/t',num2str(num_click),'.jpg']);
    g = imread(char(original_name{num_click}));
    axes(handles.axes1);
    imshow(g);

%     h_path = strcat(['./img_segmentation/t',num2str(num_click),'_seg.jpg']);
    h = imread(char(segmentation_name{num_click}));
    axes(handles.axes2);
    imshow(h);
    
    x = get(handles.edit1,'String');
    value = str2double(x);
    score_array(num_click)=value;
else
    dlmwrite('score.txt',score_array);
end


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global num_click
global segmentation_name
h = imread(char(segmentation_name{num_click}));
temp = segmentation_name{num_click};
tmp_path = split(temp,'/');
tmp=tmp_path(end);
h_path = ['./piam_img_256_select_from_kmeans/',char(tmp)];
imwrite(h,char(h_path));


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global num_click
global segmentation_name
h = imread(char(segmentation_name{num_click}));
temp = segmentation_name{num_click};
tmp_path = split(temp,'/');
tmp=tmp_path(end);
h_path = ['./piam_img_256_rest_from_kmeans/',char(tmp)];
imwrite(h,char(h_path));
