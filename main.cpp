#include <chrono>
#include <print>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/gl.h>

#include <iostream>
#include <random>
#include <vector>

static const int window_size = 1080;
static const int angle_bin_count = 256 * 4;
static const int depth_bin_count = 256 * 2;

struct vec2 {
  float x;
  float y;
};

static const char *tri_vs = R"(
#version 430 core
layout (location = 0) in vec2 a_pos;

uniform float u_time;
void main() {
    vec2 v_pos = a_pos + vec2(sin(u_time)/10.0f, cos(u_time) / 10.0f);
    gl_Position = vec4(v_pos, 0.0, 1.0);
}
)";

static const char *tri_vs_dbg = R"(
#version 430 core
layout (location = 0) in vec2 a_pos;

uniform float u_time;
void main() {
    gl_Position = vec4(a_pos + vec2(sin(u_time)/10.0f, cos(u_time) / 10.0f), 0.0, 1.0);
}
)";

static const char *tri_fs = R"(
#version 430 core
out float mask;

void main() {
  mask = 1.0;
}
)";

static const char *tri_fs_dbg = R"(
#version 430 core
out vec4 frag_color;


void main() {
  frag_color = vec4(1.0,0.0,0.0,1.0);
}
)";

static const char *mask_vs = R"(
#version 430 core
layout(location = 0) in vec2 in_position;

out vec2 v_uv;

void main() {
    v_uv = (in_position + 1.0) * 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
)";

static const char *mask_fs = R"(
#version 430 core
in vec2 v_uv;
layout(location = 0) out vec2 out_moments; 

uniform sampler2D u_clip;
uniform vec2 u_texel_size;

void main()
{
    float theta  = -v_uv.y * 2.0 * 3.14159265;
    float r_norm = v_uv.x;

    vec2 base_dir = vec2(cos(theta), sin(theta)) * r_norm;

    int samples = 2; 
    float accum  = 0.0;
    float accum2 = 0.0;

    float far_depth = 1.0;

    for (int i = -samples; i <= samples; i++)
    {
        for (int j = -samples; j <= samples; j++)
        {
            vec2 offset = vec2(i,j) * u_texel_size / max(r_norm, 0.3);

            vec2 sample_uv = (base_dir + offset + 1.0) * 0.5;
            sample_uv = clamp(sample_uv, vec2(0.0), vec2(1.0));

            float occluder = texture(u_clip, sample_uv).r;

            
            float depth = ((occluder > 0.0) ? (r_norm * r_norm) : far_depth);

            accum  += depth;
            accum2 += depth * depth;
        }
    }

    float kernel_size = float((2*samples+1)*(2*samples+1));
    float mean  = accum  / kernel_size;
    float mean2 = accum2 / kernel_size;

    out_moments = vec2(mean, mean2);
})";

static const char *quad_vs = R"(
#version 430 core
layout (location = 0) in vec2 a_pos;
out vec2 v_uv;
void main() {
    v_uv = (a_pos + 1.0) * 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
)";

static const char *quad_fs = R"(
#version 430 core
out vec4 frag_color;
in vec2 v_uv;

uniform sampler2D u_mask;
uniform int u_angle_bins;
uniform int u_depth_bins;
uniform int u_dbg;

float chebyshev_visibility(vec2 moments, float receiver)
{
    float mean  = moments.x;
    float mean2 = moments.y;

    float variance = max(mean2 - mean * mean, 0.0005);

    float d = receiver - mean;

    float p = variance / (variance + d * d);
    

    return max(p, float(receiver <= mean));
}

void main()
{
    vec2 centered = v_uv * 2.0 - 1.0;
    centered.x *= -1.0;

    float r = length(centered);
    float theta = atan(centered.y, centered.x);

    float theta_norm =
        float(u_angle_bins) *
        (theta + 3.14159265) /
        (2.0 * 3.14159265);

    float angle_index =
        clamp(theta_norm, 0, u_angle_bins - 1);

    float r_norm = r;

    float max_depth_bin =
        clamp(r_norm * float(u_depth_bins),
              0.0,
              float(u_depth_bins - 1));

    float visibility = 1.0;   

    float dr = 1.0 / float(u_depth_bins);
    if (u_dbg == 0)
    {
        for (int d = int(max_depth_bin); d >= 0; --d)
        {
            float current_r = float(d) / float(u_depth_bins);
            vec2 uv = vec2(
                current_r,
                angle_index / float(u_angle_bins)
            );

            vec2 moments = texture(u_mask, uv).rg;

            float v = chebyshev_visibility(moments, 0.8);

            
            visibility *= v;
        }
    }
    else
    {
        vec2 moments = texture(u_mask, v_uv).rg;
        frag_color = vec4(moments.x, moments.y, 0.0, 1.0);
        return;
    }

    r_norm = max(r_norm, 0.001) * 5;
    float light = max(0.1 / (r_norm * r_norm), 0.1);
    float shadow = light * visibility;

    frag_color = vec4(vec3(shadow), 1.0);
}
)";

GLuint compile_shader(GLenum type, const char *src) {
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &src, nullptr);
  glCompileShader(shader);

  GLint ok;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetShaderInfoLog(shader, 1024, nullptr, log);
    std::cerr << log << std::endl;
  }

  return shader;
}

GLuint create_program(const char *vs, const char *fs) {
  std::println("Compiling shader program...");
  GLuint p = glCreateProgram();
  GLuint sv = compile_shader(GL_VERTEX_SHADER, vs);
  GLuint sf = compile_shader(GL_FRAGMENT_SHADER, fs);

  glAttachShader(p, sv);
  glAttachShader(p, sf);
  glLinkProgram(p);

  glDeleteShader(sv);
  glDeleteShader(sf);

  return p;
}

int main() {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  // glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);

  GLFWwindow *window = glfwCreateWindow(
      window_size, window_size, "2D Angular Shadow Mask", nullptr, nullptr);

  glfwMakeContextCurrent(window);
  gladLoadGL(glfwGetProcAddress);
  glfwSwapInterval(0);

  // Mask FBO and texture
  GLuint mask_tex, mask_fbo;
  glGenFramebuffers(1, &mask_fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, mask_fbo);

  glGenTextures(1, &mask_tex);
  glBindTexture(GL_TEXTURE_2D, mask_tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, depth_bin_count, angle_bin_count, 0,
               GL_RG, GL_FLOAT, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         mask_tex, 0);

  GLenum draw_buffers[1] = {GL_COLOR_ATTACHMENT0};
  glDrawBuffers(1, draw_buffers);

  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    std::cout << "mask_fbo incomplete!\n";

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  GLuint clip_tex, clip_fbo;
  glGenFramebuffers(1, &clip_fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, clip_fbo);

  glGenTextures(1, &clip_tex);
  glBindTexture(GL_TEXTURE_2D, clip_tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, depth_bin_count * 8,
               depth_bin_count * 8, 0, GL_RED, GL_FLOAT, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         clip_tex, 0);

  glDrawBuffers(1, draw_buffers);

  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    std::cout << "clip_fbo incomplete!\n";

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  GLuint tri_prog = create_program(tri_vs, tri_fs);
  GLuint mask_prog = create_program(mask_vs, mask_fs);
  GLuint tri_prog_dbg = create_program(tri_vs_dbg, tri_fs_dbg);
  GLuint quad_prog = create_program(quad_vs, quad_fs);

  std::vector<vec2> tris;
  std::random_device rng_dev;
  std::mt19937 rng(rng_dev());
  std::uniform_real_distribution<float> dist(-0.8f, 0.8f);

  for (int i = 0; i < 5; ++i) {
    float cx = dist(rng);
    float cy = dist(rng);

    float dx = dist(rng) / 2.0f;
    float dy = dist(rng) / 2.0f;

    tris.push_back({cx, cy});
    tris.push_back({cx + dx, cy + dy});
    tris.push_back({cx - dy, cy - dx});
  }

  GLuint tri_vao, tri_vbo;
  glGenVertexArrays(1, &tri_vao);
  glGenBuffers(1, &tri_vbo);

  glBindVertexArray(tri_vao);
  glBindBuffer(GL_ARRAY_BUFFER, tri_vbo);
  glBufferData(GL_ARRAY_BUFFER, tris.size() * sizeof(vec2), tris.data(),
               GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), (void *)0);
  glEnableVertexAttribArray(0);

  float quad[] = {-1, -1, 1, -1, 1, 1, -1, 1};
  GLuint quad_vao, quad_vbo;
  glGenVertexArrays(1, &quad_vao);
  glGenBuffers(1, &quad_vbo);

  glBindVertexArray(quad_vao);
  glBindBuffer(GL_ARRAY_BUFFER, quad_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);

  glDisable(GL_BLEND);

  float dt = 1.0f;
  while (!glfwWindowShouldClose(window)) {
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
      std::println("dt: {:.2f}ms", dt * 1000);
    }
    auto frame_start = std::chrono::steady_clock::now();
    static float t = 0;
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) {
      t += dt;
    }
    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS) {
      t -= dt;
    }

    // begin shadow pass
    glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);
    glBindFramebuffer(GL_FRAMEBUFFER, clip_fbo);
    glViewport(0, 0, depth_bin_count * 8, depth_bin_count * 8);

    GLuint clear_val = 0;
    glClearBufferuiv(GL_COLOR, 0, &clear_val);

    glUseProgram(tri_prog);
    glBindVertexArray(tri_vao);
    glDrawArrays(GL_TRIANGLES, 0, tris.size());
    glUniform1f(glGetUniformLocation(tri_prog, "u_time"), t);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);

    glBindFramebuffer(GL_FRAMEBUFFER, mask_fbo);
    glViewport(0, 0, depth_bin_count, angle_bin_count);

    glClearBufferuiv(GL_COLOR, 0, &clear_val);
    glUseProgram(mask_prog);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, clip_tex);
    glUniform1i(glGetUniformLocation(mask_prog, "u_clip"), 0);

    glUniform1i(glGetUniformLocation(mask_prog, "u_angle_bins"),
                angle_bin_count);

    glUniform2f(glGetUniformLocation(mask_prog, "u_texel_size"),
                1.0f / (depth_bin_count * 8), 1.0f / (depth_bin_count * 8));

    glBindVertexArray(quad_vao);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // end shadow pass

    glViewport(0, 0, window_size, window_size);
    glUseProgram(quad_prog);

    glUniform1i(glGetUniformLocation(quad_prog, "u_angle_bins"),
                angle_bin_count);
    glUniform1i(glGetUniformLocation(quad_prog, "u_depth_bins"),
                depth_bin_count);
    glUniform1i(glGetUniformLocation(quad_prog, "u_dbg"),
                glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mask_tex);
    glUniform1i(glGetUniformLocation(quad_prog, "u_mask"), 0);

    glBindVertexArray(quad_vao);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    if (glfwGetKey(window, GLFW_KEY_R) != GLFW_PRESS) {
      glUseProgram(tri_prog_dbg);
      glUniform1f(glGetUniformLocation(tri_prog_dbg, "u_time"), t);

      glBindVertexArray(tri_vao);
      glDrawArrays(GL_TRIANGLES, 0, tris.size());
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
    dt = std::chrono::duration<float>(std::chrono::steady_clock::now() -
                                      frame_start)
             .count();
  }

  glfwTerminate();
  return 0;
}
