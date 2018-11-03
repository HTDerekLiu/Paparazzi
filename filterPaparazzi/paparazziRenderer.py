from include import *

class paparazziRenderer:
	def __init__(self, imgSize = 256):
		self.imgSize = imgSize # number of pixel alogn one axis (squared image) 
		self.window = None # handle for the window of this renderer
		self.mainShader = None
		self.quadShader = None

		self.window = None
		self.vao = None
		self.vbo_V = None
		self.vbo_N = None
		self.vbo_FIdx = None
		self.data_V = None
		self.data_N = None
		self.data_FIdx = None

		self.vao_quad = None
		self.vbo_quad_V = None
		self.vbo_quad_UV = None
		self.data_quad_V = None
		self.data_quad_UV = None

		self.fbo = None
		self.texture = None
		self.rbo = None

		self.mvp = None # MVP matrix (orthographic projection)
		self.rgb = None # store rendered image

		# direction and color of the three directional lights 
		self.L0Dir = np.array([0,0,0], dtype = np.float32) 
		self.L1Dir = np.array([0,0,0], dtype = np.float32)
		self.L2Dir = np.array([0,0,0], dtype = np.float32)
		self.L0Color = np.array([0,0,0], dtype = np.float32)
		self.L1Color = np.array([0,0,0], dtype = np.float32)
		self.L2Color = np.array([0,0,0], dtype = np.float32)

		vertexShader1 = """#version 330
		layout(location = 0) in vec3 vert;
		layout(location = 1) in vec3 N;
		layout(location = 2) in vec3 FIdx;
		uniform mat4 MVP;
		out vec3 outN;
		out vec3 outFIdx;
		void main () {
		    outN = N;
		    outFIdx = FIdx;
		    gl_Position = MVP*vec4(vert, 1.0f);
		}"""

		fragmentShader1 = """#version 330
		in vec3 outN;
		in vec3 outFIdx;
		uniform vec3 L0Dir;
		uniform vec3 L1Dir;
		uniform vec3 L2Dir;
		uniform vec3 L0Color;
		uniform vec3 L1Color;
		uniform vec3 L2Color;
		uniform int renderMode;
		out vec4 color;
		void main () {
			if (renderMode == 0){ // default
				vec3 N = normalize(outN);
				vec3 RGB = dot(L0Dir, N)*L0Color + dot(L1Dir, N)*L1Color + dot(L2Dir, N)*L2Color;
		    	color = vec4((RGB.x+1)/2, (RGB.y+1)/2, (RGB.z+1)/2, 1.);
			}
			if (renderMode == 1){ // render surface normal
				color = vec4(normalize(outN), 1.0f);
			}
			if (renderMode == 2){ // render face indices
				color = vec4(outFIdx, 1.0f);
			}
			if (renderMode == 3){ // draw mask
				vec3 N = normalize(outN);
				vec3 RGB = dot(L0Dir, N)*L0Color + dot(L1Dir, N)*L1Color + dot(L2Dir, N)*L2Color;
		    	color = vec4(RGB, 1.);
			}
		}"""

		vertexShader2 = """# version 330
		layout(location=3) in vec2 position;
		layout(location=4) in vec2 texCoords;
		out vec2 TexCoords;
		void main()
		{
		    gl_Position = vec4(position.x, position.y, 0.0f, 1.0f);
		    TexCoords = texCoords;
		}
		"""

		fragmentShader2 = """# version 330
		in vec2 TexCoords;
		out vec4 color;
		uniform sampler2D screenTexture;
		void main()
		{
		    vec3 backColor = vec4(texture(screenTexture, TexCoords)).xyz; // original rendered pixel color value
		    color = vec4(backColor, 1.0);
		}
		"""

		if not glfwInit():
			print("Failed to initialize GLFW")
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
		glfwWindowHint(GLFW_VISIBLE, GL_FALSE)

		self.window = glfwCreateWindow(self.imgSize, self.imgSize, "VAO Test", None, None)
		glfwMakeContextCurrent(self.window)
		glfwSwapInterval(0) # disable vsync

		self.vao = GLuint()
		glGenVertexArrays(1, self.vao)
		glBindVertexArray(self.vao)
		self.vbo_V = GLuint() # store vertex positions
		glGenBuffers(1, self.vbo_V)
		glBindBuffer(GL_ARRAY_BUFFER, self.vbo_V)
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.float_size(3), self.pointer_offset(0))
		self.vbo_N = GLuint()
		glGenBuffers(1, self.vbo_N) # store normals
		glBindBuffer(GL_ARRAY_BUFFER, self.vbo_N)
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, self.float_size(3), self.pointer_offset(0)) 
		self.vbo_FIdx = GLuint()
		glGenBuffers(1, self.vbo_FIdx) # store face indices
		glBindBuffer(GL_ARRAY_BUFFER, self.vbo_FIdx)
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, self.float_size(3), self.pointer_offset(0))   
		glEnableVertexAttribArray(0)
		glEnableVertexAttribArray(1)
		glEnableVertexAttribArray(2)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		glBindVertexArray(0)

		self.vao_quad = GLuint()
		glGenVertexArrays(1, self.vao_quad)
		glBindVertexArray(self.vao_quad)
		self.vbo_quad_V = GLuint() 
		glGenBuffers(1, self.vbo_quad_V)
		glBindBuffer(GL_ARRAY_BUFFER, self.vbo_quad_V)
		glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, self.float_size(2), self.pointer_offset(0))
		self.vbo_quad_UV = GLuint() 
		glGenBuffers(1, self.vbo_quad_UV)
		glBindBuffer(GL_ARRAY_BUFFER, self.vbo_quad_UV)
		glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, self.float_size(2), self.pointer_offset(0))
		glEnableVertexAttribArray(3)
		glEnableVertexAttribArray(4)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		glBindVertexArray(0)

		self.mainShader = self.create_shader(vertexShader1, fragmentShader1)
		self.quadShader = self.create_shader(vertexShader2, fragmentShader2)

		self.fbo = GLuint()
		glGenFramebuffers(1, self.fbo)
		glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
		self.texture = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, self.texture)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, self.imgSize, self.imgSize, 0, GL_RGB, GL_FLOAT, None) #'bad', 32F, GL FLOAT
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture, 0)
		self.rbo = GLuint()
		glGenRenderbuffers(1,self.rbo)
		glBindRenderbuffer(GL_RENDERBUFFER, self.rbo)
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, self.imgSize, self.imgSize)
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self.rbo)
		if (GL_FRAMEBUFFER_COMPLETE != glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER)):
			print "fbo_read incomplete"

		glBindFramebuffer(GL_FRAMEBUFFER, 0)
		glBindTexture(GL_TEXTURE_2D, 0)
		glBindRenderbuffer(GL_RENDERBUFFER, 0)
		glDepthFunc(GL_LESS)
		# print "openGL version:", glGetString(GL_VERSION) 

		self.setQuad()

	# =================================
	# renderer functions
	# =================================
	def draw(self, renderMode = "default"):
		# The main draw function
		# 
		# Input
		#   renderMode: a string 
		#     "normal" renders surface normal image
		#     "faceIdx" renders image of face indices
		#     other strings use default renderer which uses directional lights
		#
		# Output
		# 	rendered image
		glfwMakeContextCurrent(self.window)
		w, h = glfwGetWindowSize(self.window)
		glViewport(0,0, w, h) # for retina screen

		glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
		glEnable(GL_DEPTH_TEST)
		glClearColor(0.0, 0.0, 0.0, 1.0)
		glClearDepth(1.0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

		glBindVertexArray(self.vao)
		glUseProgram(self.mainShader)

		# set rendering modes
		if renderMode == "normal":
			glUniform1i(glGetUniformLocation(self.mainShader,"renderMode"), 1)
		elif renderMode == "faceIdx":
			glUniform1i(glGetUniformLocation(self.mainShader,"renderMode"), 2)
		else:
			glUniform1i(glGetUniformLocation(self.mainShader,"renderMode"), 0)

		glUniformMatrix4fv(glGetUniformLocation(self.mainShader, "MVP"), 1, GL_FALSE, self.mvp)
		glUniform3f(glGetUniformLocation(self.mainShader,"L0Dir"), self.L0Dir[0], self.L0Dir[1], self.L0Dir[2])
		glUniform3f(glGetUniformLocation(self.mainShader,"L1Dir"), self.L1Dir[0], self.L1Dir[1], self.L1Dir[2])
		glUniform3f(glGetUniformLocation(self.mainShader,"L2Dir"), self.L2Dir[0], self.L2Dir[1], self.L2Dir[2])
		glUniform3f(glGetUniformLocation(self.mainShader,"L0Color"), self.L0Color[0], self.L0Color[1], self.L0Color[2])
		glUniform3f(glGetUniformLocation(self.mainShader,"L1Color"), self.L1Color[0], self.L1Color[1], self.L1Color[2])
		glUniform3f(glGetUniformLocation(self.mainShader,"L2Color"), self.L2Color[0], self.L2Color[1], self.L2Color[2])

		glDrawArrays(GL_TRIANGLES, 0, self.data_V.shape[0]) # draw mesh
		rgb = glReadPixels(0,0, self.imgSize, self.imgSize, GL_RGB, GL_FLOAT, outputType=None)
		glBindVertexArray(0)
		glBindFramebuffer(GL_FRAMEBUFFER, 0)
		glBindTexture(GL_TEXTURE_2D, 0)

		if renderMode == "faceIdx":
			rgb[:,:,0] *= self.drawNonBackgroundMask()
			rgb[:,:,1] *= self.drawNonBackgroundMask()
			rgb[:,:,2] *= self.drawNonBackgroundMask()
			idxImg = rgb.astype(int) - 1
			visIdxList = idxImg.reshape((idxImg.shape[0]*idxImg.shape[1], idxImg.shape[2]))
			emptyRowIdx = np.unique(np.where(visIdxList == -1)[0])
			visIdxList = np.delete(visIdxList, emptyRowIdx, axis = 0)
			visRow = np.array(range(self.imgSize**2))
			visRow = np.delete(visRow, emptyRowIdx)
			return idxImg, visIdxList[:,0], visRow
		else:
			self.rgb = rgb
			return rgb

	def setMesh(self, V, F, shading = "flat"):
		# Set the mesh data
		#
		# Input
		#   V: (#V, 3) numpy array, vertex positions
		#   F: (#F, 3) numpy array, face indices
		#   shading: a string, "flat" or "gouraud"
		#
		# Notes:
		# 	the gradient computation only supports "flat" shading, thus
		#   "gouraud" shading can only be used for visualization
		glfwMakeContextCurrent(self.window)
		out = V[F.reshape(F.shape[0]*F.shape[1]),:]
		self.data_V = out.astype(np.float32)
		glBindBuffer(GL_ARRAY_BUFFER, self.vbo_V)
		glBufferData(GL_ARRAY_BUFFER, self.data_V, GL_STATIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)

		if shading == "flat":
			N = faceNormals(V,F)
			out = np.tile(N, [1,3]).reshape(F.shape[0]*F.shape[1], 3)
		elif shading == "gouraud":
			N = vertexNormals(V,F)
			out = N[F.reshape(F.shape[0]*F.shape[1]),:]
		self.data_N = out.astype(np.float32)
		glBindBuffer(GL_ARRAY_BUFFER, self.vbo_N)
		glBufferData(GL_ARRAY_BUFFER, self.data_N, GL_STATIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)

		faceIdx = np.tile(np.array(range(F.shape[0]))[:,None], (1,3)) + 1
		out = np.tile(faceIdx, [1,3])
		out = out.reshape(F.shape[0]*F.shape[1], 3)
		self.data_FIdx = out.astype(np.float32)
		glBindBuffer(GL_ARRAY_BUFFER, self.vbo_FIdx)
		glBufferData(GL_ARRAY_BUFFER, self.data_FIdx, GL_STATIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)

	def setCamera(self, windowSize, viewCenter, viewDir, nearPlane = 0, farPlane = 100):
		# Set the camera parameters, specifically the MVP matrix
		#
		# Input
		#   fieldOfView: a float, field of view
		#   cameraCenter: (3,) numpy array, center of the camera in the world coordinate
		#   cameraDir: (3,) numpy array, the direction where the camera is facing
		#   nearPlane: a float, near plane position (coordinate value of the z-axis of the camera)
		#   farPlane: a float, far plane position (coordinate value of the z-axis of the camera)

		# orthographic projection matrix
		bbox = np.array([-windowSize,windowSize,-windowSize,windowSize,nearPlane,farPlane]).astype(np.float32)
		P = self.getOrthoProj(bbox)
		# translation matrix
		T = np.identity(4)
		T[:3, 3] = -viewCenter
		# rotation marix
		cameraDir = viewDir / np.linalg.norm(viewDir)
		cameraDir[2] *= -1.0
		R = self.getRotationMatrix(np.array([0,0,1]),cameraDir)
		mvp = P.dot(R).dot(T).T
		self.mvp = mvp

	def getCameraFrame(self):
		# get camera frame axes
		#
		# Output
		# 	local frame axes (as 3 unit vectors)
		lx = self.mvp[:3,0].flatten()
		ly = self.mvp[:3,1].flatten()
		lz = -self.mvp[:3,2].flatten()
		# make sure they are normalized
		lx = lx / np.linalg.norm(lx)
		ly = ly / np.linalg.norm(ly)
		lz = lz / np.linalg.norm(lz)
		return lx, ly, lz

	def setLights(self, L0Dir = None, L1Dir = None, L2Dir = None,\
				L0Color = None, L1Color = None, L2Color = None):
		# set direction (in global frame) and color of the three directional lights
		# 
		# Input
		# 	L0Dir: (3,) numpy vector, direction of light 0 
		# 	L1Dir: (3,) numpy vector, direction of light 1 
		# 	L2Dir: (3,) numpy vector, direction of light 2 
		# 	L0Color: (3,) numpy vector, color of light 0 (each value is between 0 and 1)
		# 	L1Color: (3,) numpy vector, color of light 1 (each value is between 0 and 1)
		# 	L2Color: (3,) numpy vector, color of light 2 (each value is between 0 and 1)
		if L0Dir is not None:
			L0Dir = L0Dir / np.linalg.norm(L0Dir)
			self.L0Dir = L0Dir.astype(np.float32)
		if L1Dir is not None:
			L1Dir = L1Dir / np.linalg.norm(L1Dir)
			self.L1Dir = L1Dir.astype(np.float32)
		if L2Dir is not None:
			L2Dir = L2Dir / np.linalg.norm(L2Dir)
			self.L2Dir = L2Dir.astype(np.float32)
		if L0Color is not None:
			self.L0Color = L0Color.astype(np.float32)
		if L1Color is not None:
			self.L1Color = L1Color.astype(np.float32)
		if L2Color is not None:
			self.L2Color = L2Color.astype(np.float32)

	def getLights(self):
		# get renderer's lighting
		#
		# Output
		#	lightDir: (3,3) numpy matrix where lightDir[:,0] is light 0 direction and so on
		#	lightColor: (3,3) numpy matrix where lightColor[:,0] is light 0 color and so on
		lightDir = np.concatenate((self.L0Dir[None,:], self.L1Dir[None,:], self.L2Dir[None,:]), axis = 0)
		lightColor = np.concatenate((self.L0Color[None,:], self.L1Color[None,:], self.L2Color[None,:]), axis = 0)
		return lightDir, lightColor

	# =================================
	# utility functions
	# =================================
	def float_size(self, n=1):
		return sizeof(ctypes.c_float) * n

	def int_size(self, n=1):
		return sizeof(ctypes.c_int) * n

	def pointer_offset(self, n=0):
		return ctypes.c_void_p(self.float_size(n))

	def close(self):
		glfwTerminate()

	def create_shader(self, vertex_shader, fragment_shader):
		vs_id = GLuint(glCreateShader(GL_VERTEX_SHADER))  # shader id for vertex shader
		glShaderSource(vs_id, [vertex_shader], None)  # Send the code of the shader
		glCompileShader(vs_id)  # compile the shader code
		status = glGetShaderiv(vs_id, GL_COMPILE_STATUS)
		if status != 1:
			print('VERTEX SHADER ERROR')
			print(glGetShaderInfoLog(vs_id).decode())
		fs_id = GLuint(glCreateShader(GL_FRAGMENT_SHADER))
		glShaderSource(fs_id, [fragment_shader], None)
		glCompileShader(fs_id)
		status = glGetShaderiv(fs_id, GL_COMPILE_STATUS)
		if status != 1:
			print('FRAGMENT SHADER ERROR')
			print(glGetShaderInfoLog(fs_id).decode())
		# Link the shaders into a single program
		program_id = GLuint(glCreateProgram())
		glAttachShader(program_id, vs_id)
		glAttachShader(program_id, fs_id)
		glLinkProgram(program_id)
		status = glGetProgramiv(program_id, GL_LINK_STATUS)
		if status != 1:
			print('status', status)
			print('SHADER PROGRAM', glGetShaderInfoLog(program_id))
		glDeleteShader(vs_id)
		glDeleteShader(fs_id)
		return program_id

	def setQuad(self):
		glfwMakeContextCurrent(self.window)
		self.data_quad_V = np.array([\
			[-1.0, 1.0],[-1.0, -1.0],[1.0, -1.0],\
			[-1.0, 1.0],[1.0, -1.0],[1.0, 1.0]], dtype=np.float32)
		glBindBuffer(GL_ARRAY_BUFFER, self.vbo_quad_V)
		glBufferData(GL_ARRAY_BUFFER, self.data_quad_V, GL_STATIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)

		self.data_quad_UV = np.array([\
			[0.0, 1.0],[0.0, 0.0],[1.0, 0.0],\
			[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]], dtype=np.float32)
		glBindBuffer(GL_ARRAY_BUFFER, self.vbo_quad_UV)
		glBufferData(GL_ARRAY_BUFFER, self.data_quad_UV, GL_STATIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)

	def getOrthoProj(self,bbox):
		l,r,b,t,n,f = bbox
		M = np.identity(4)
		M[0,0] = 2 / (r-l)
		M[1,1] = 2 / (t-b)
		M[2,2] = -2 / (f-n)
		M[0,3] = -(r+l) / (r-l)
		M[1,3] = -(t+b) / (t-b)
		M[2,3] = -(f+n) / (f-n)
		return M

	def getRotationMatrix(self,v1, v2):
		v1 = v1/ np.linalg.norm(v1)
		v2 = v2 / np.linalg.norm(v2)
		uvw = np.cross(v1, v2)
		rcos = np.dot(v1, v2)
		rsin = np.linalg.norm(uvw)
		if not np.isclose(rsin, 0):
			uvw /= rsin
		u, v, w = uvw
		R33 = (rcos * np.eye(3)+rsin \
			* np.array([[ 0, -w,  v],[ w,  0, -u],[-v,  u,  0]]) \
			+(1.0 - rcos) * uvw[:,None] * uvw[None,:])
		R44 = np.identity(4)
		R44[:3,:3] = R33
		return R44

	def drawNonBackgroundMask(self):
		# Output
		#   non-background pixel indices (need to "setMesh" first)
		glfwMakeContextCurrent(self.window)
		w, h = glfwGetWindowSize(self.window)
		glViewport(0,0, w, h) # for retina screen

		glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
		glEnable(GL_DEPTH_TEST)
		glClearColor(0.0, 0.0, 0.0, 1.0)
		glClearDepth(1.0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

		glBindVertexArray(self.vao)
		glUseProgram(self.mainShader)
		glUniform1i(glGetUniformLocation(self.mainShader,"renderMode"), 3)

		x,y,z = self.getCameraFrame()
		glUniformMatrix4fv(glGetUniformLocation(self.mainShader, "MVP"), 1, GL_FALSE, self.mvp)

		glUniform3f(glGetUniformLocation(self.mainShader,"L0Dir"), z[0], z[1], z[2])
		glUniform3f(glGetUniformLocation(self.mainShader,"L1Dir"), 0,0,0)
		glUniform3f(glGetUniformLocation(self.mainShader,"L2Dir"), 0,0,0)
		glUniform3f(glGetUniformLocation(self.mainShader,"L0Color"), 1.0,1.0,1.0)
		glUniform3f(glGetUniformLocation(self.mainShader,"L1Color"), 0,0,0)
		glUniform3f(glGetUniformLocation(self.mainShader,"L2Color"), 0,0,0)

		glDrawArrays(GL_TRIANGLES, 0, self.data_V.shape[0]) # draw mesh
		rgb = glReadPixels(0,0, self.imgSize, self.imgSize, GL_RGB, GL_FLOAT, outputType=None)
		glBindVertexArray(0)
		glBindFramebuffer(GL_FRAMEBUFFER, 0)
		glBindTexture(GL_TEXTURE_2D, 0)

		rgb = rgb[:,:,0]
		rgb[np.where(rgb<=0)] = 0.0
		rgb[np.where(rgb> 0)] = 1.0
		return rgb




