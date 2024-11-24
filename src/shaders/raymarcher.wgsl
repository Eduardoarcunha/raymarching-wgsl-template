const THREAD_COUNT = 16;
const PI = 3.1415927f;
const MAX_DIST = 1000.0;

@group(0) @binding(0)  
  var<storage, read_write> fb : array<vec4f>;

@group(1) @binding(0)
  var<storage, read_write> uniforms : array<f32>;

@group(2) @binding(0)
  var<storage, read_write> shapesb : array<shape>;

@group(2) @binding(1)
  var<storage, read_write> shapesinfob : array<vec4f>;

struct shape {
  transform : vec4f, // xyz = position
  radius : vec4f, // xyz = scale, w = global scale
  rotation : vec4f, // xyz = rotation
  op : vec4f, // x = operation, y = k value, z = repeat mode, w = repeat offset
  color : vec4f, // xyz = color
  animate_transform : vec4f, // xyz = animate position value (sin amplitude), w = animate speed
  animate_rotation : vec4f, // xyz = animate rotation value (sin amplitude), w = animate speed
  quat : vec4f, // xyzw = quaternion
  transform_animated : vec4f, // xyz = position buffer
};

struct march_output {
  color : vec3f,
  depth : f32,
  outline : bool,
};

fn op_smooth_union(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  var k_eps = max(k, 0.0001);
  return vec4f(col1, d1);
}

fn op_smooth_subtraction(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  var k_eps = max(k, 0.0001);
  return vec4f(col1, d1);
}

fn op_smooth_intersection(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  var k_eps = max(k, 0.0001);
  return vec4f(col1, d1);
}

fn op(op: f32, d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  // union
  if (op < 1.0)
  {
    return op_smooth_union(d1, d2, col1, col2, k);
  }

  // subtraction
  if (op < 2.0)
  {
    return op_smooth_subtraction(d2, d1, col2, col1, k);
  }

  // intersection
  return op_smooth_intersection(d2, d1, col2, col1, k);
}

fn repeat(p: vec3f, offset: vec3f) -> vec3f 
{
  if (offset.x == 0.0 || offset.y == 0.0 || offset.z == 0.0)
  {
    return p;
  }
    // Create offset vector for centering
    var half_offset = 0.5 * offset;    
    return modc(p + half_offset, offset) - half_offset;
}

fn transform_p(p: vec3f, option: vec2f) -> vec3f 
{
    // Normal mode (no transformation)
    if (option.x <= 1.0) {
        return p;
    }
    
    // Repeat mode
    // option.x > 1.0 indicates repeat mode is enabled
    // option.y contains the repeat offset (same for all axes)
    return repeat(p, vec3f(option.y));
}

fn scene(p: vec3f) -> vec4f // xyz = color, w = distance
{
    var d = mix(100.0, p.y, uniforms[17]);  // Initial floor distance if enabled

    var spheresCount = i32(uniforms[2]);
    var boxesCount = i32(uniforms[3]);
    var torusCount = i32(uniforms[4]);

    var all_objects_count = spheresCount + boxesCount + torusCount;
    var result = vec4f(vec3f(1.0), d);

    for (var i = 0; i < all_objects_count; i = i + 1)
    {
        // Get the shape info for current object
        var shape_info = shapesinfob[i];
        var shape_type = shape_info.x;
        var shape_index = i32(shape_info.y);
        
        // Get the actual shape data
        var shape_data = shapesb[shape_index];

        var _quat = quaternion_from_euler(shape_data.rotation.xyz);

        
        // Transform the point relative to shape position
        var transformed_p = p - (shape_data.transform.xyz + shape_data.transform_animated.xyz);
        
        // Apply repeat transformation if enabled
        transformed_p = transform_p(transformed_p, shape_data.op.zw);
        
        // Variable to store the distance for this shape
        var shape_distance = 0.0;
        
        // Calculate distance based on shape type
        if (shape_type < 1.0)
        {
          shape_distance = sdf_sphere(transformed_p, shape_data.radius, _quat);
        } else if (shape_type < 2.0)   {
          shape_distance = sdf_round_box(transformed_p, shape_data.radius.xyz, shape_data.radius.w, _quat);
        } else if (shape_type < 3.0)  {
          shape_distance = sdf_torus(transformed_p, shape_data.radius.xy,  _quat);
        } else {
          shape_distance = MAX_DIST;
        }

        let res = vec4f(shape_data.color.xyz,shape_distance);
        result = res; // assign color and distance 
        
    }

    return result;
}

fn march(ro: vec3f, rd: vec3f) -> march_output
{
  var max_marching_steps = i32(uniforms[5]);
  var EPSILON = uniforms[23];

  var depth = 0.0;
  var color = vec3f(0.0);
  var march_step = uniforms[22];
  
  for (var i = 0; i < max_marching_steps; i = i + 1)
  {
      // Get current position along the ray
      var current = ro + rd * depth;
      
      // Get the scene distance at this point
      var scene_res = scene(current);
      var dist = scene_res.w;
      
      // Check if we've hit something
      if (dist < EPSILON) {
          // We hit something! Return the color and depth
          color = scene_res.xyz;
          return march_output(color, depth, false);
      }
      
      // Check if we've gone too far
      if (depth > MAX_DIST) {
          // We didn't hit anything, return background
          return march_output(color, depth, false);
      }
      
      // Move along the ray by the safe distance (dist)
      depth += dist * march_step;
  }

  // If we run out of steps, return background
  return march_output(color, depth, false);
}

fn get_normal(p: vec3f) -> vec3f 
{
    let eps = uniforms[23]; // Using your epsilon value from uniforms
    let k = vec2f(1.0, -1.0);
    
    return normalize(k.xyy * scene(p + k.xyy * eps).w + 
                    k.yyx * scene(p + k.yyx * eps).w + 
                    k.yxy * scene(p + k.yxy * eps).w + 
                    k.xxx * scene(p + k.xxx * eps).w);
}

// https://iquilezles.org/articles/rmshadows/
fn get_soft_shadow(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, k: f32) -> f32 
{
    // Initialize shadow multiplier
    var res = 1.0;
    
    // Initialize distance traveled
    var t = tmin;
    
    // March along shadow ray
    for(var i = 0; i < 32; i = i + 1) {  // Using fixed iterations for stability
        if(t >= tmax) { break; }
        
        // Get distance to scene
        let h = scene(ro + rd * t).w;
        
        // Early exit if we hit something
        if(h < uniforms[23]) { return 0.0; }
        
        // Calculate shadow softness
        res = min(res, k * h / t);
        
        // Move along ray
        t += h;
    }
    
    // Clamp and return result
    return clamp(res, 0.0, 1.0);
}

fn get_AO(current: vec3f, normal: vec3f) -> f32
{
  var occ = 0.0;
  var sca = 1.0;
  for (var i = 0; i < 5; i = i + 1)
  {
    var h = 0.001 + 0.15 * f32(i) / 4.0;
    var d = scene(current + h * normal).w;
    occ += (h - d) * sca;
    sca *= 0.95;
  }

  return clamp( 1.0 - 2.0 * occ, 0.0, 1.0 ) * (0.5 + 0.5 * normal.y);
}

fn get_ambient_light(light_pos: vec3f, sun_color: vec3f, rd: vec3f) -> vec3f
{
  var backgroundcolor1 = int_to_rgb(i32(uniforms[12]));
  var backgroundcolor2 = int_to_rgb(i32(uniforms[29]));
  var backgroundcolor3 = int_to_rgb(i32(uniforms[30]));
  
  var ambient = backgroundcolor1 - rd.y * rd.y * 0.5;
  ambient = mix(ambient, 0.85 * backgroundcolor2, pow(1.0 - max(rd.y, 0.0), 4.0));

  var sundot = clamp(dot(rd, normalize(vec3f(light_pos))), 0.0, 1.0);
  var sun = 0.25 * sun_color * pow(sundot, 5.0) + 0.25 * vec3f(1.0,0.8,0.6) * pow(sundot, 64.0) + 0.2 * vec3f(1.0,0.8,0.6) * pow(sundot, 512.0);
  ambient += sun;
  ambient = mix(ambient, 0.68 * backgroundcolor3, pow(1.0 - max(rd.y, 0.0), 16.0));

  return ambient;
}

fn get_light(current: vec3f, obj_color: vec3f, rd: vec3f) -> vec3f 
{
    var light_position = vec3f(uniforms[13], uniforms[14], uniforms[15]);
    var sun_color = int_to_rgb(i32(uniforms[16]));
    var ambient = get_ambient_light(light_position, sun_color, rd);
    var normal = get_normal(current);

    // If the object is too far, return ambient
    if (length(current) > uniforms[20] + uniforms[8]) {
        return ambient;
    }

    // Calculate light direction and distance
    var light_dir = normalize(light_position - current);
    var light_dist = length(light_position - current);

    // Diffuse lighting
    var diff = max(dot(normal, light_dir), 0.0);
    
    // Specular lighting
    var ref_dir = reflect(-light_dir, normal);
    
    // Shadow calculation
    var shadow = get_soft_shadow(current, light_dir, 0.1, light_dist, 32.0);
    
    // Ambient occlusion
    var ao = get_AO(current, normal);
    
    // Combine all lighting components
    var diffuse = obj_color * sun_color * diff;
    
    return (ambient * obj_color + (diffuse) * shadow) * ao;
}

fn set_camera(ro: vec3f, ta: vec3f, cr: f32) -> mat3x3<f32>
{
  var cw = normalize(ta - ro);
  var cp = vec3f(sin(cr), cos(cr), 0.0);
  var cu = normalize(cross(cw, cp));
  var cv = normalize(cross(cu, cw));
  return mat3x3<f32>(cu, cv, cw);
}

fn animate(val: vec3f, time_scale: f32, offset: f32) -> vec3f
{
  return vec3f(0.0);
}

@compute @workgroup_size(THREAD_COUNT, 1, 1)
fn preprocess(@builtin(global_invocation_id) id : vec3u)
{
  var time = uniforms[0];
  var spheresCount = i32(uniforms[2]);
  var boxesCount = i32(uniforms[3]);
  var torusCount = i32(uniforms[4]);
  var all_objects_count = spheresCount + boxesCount + torusCount;

  if (id.x >= u32(all_objects_count))
  {
    return;
  }
  return;

  // optional: performance boost
  // Do all the transformations here and store them in the buffer since this is called only once per object and not per pixel
}

@compute @workgroup_size(THREAD_COUNT, THREAD_COUNT, 1)
fn render(@builtin(global_invocation_id) id : vec3u)
{
    // unpack data
    var fragCoord = vec2f(f32(id.x), f32(id.y));
    var rez = vec2(uniforms[1]);
    var time = uniforms[0];

    // camera setup
    var lookfrom = vec3(uniforms[6], uniforms[7], uniforms[8]);
    var lookat = vec3(uniforms[9], uniforms[10], uniforms[11]);
    var camera = set_camera(lookfrom, lookat, 0.0);
    var ro = lookfrom;

    // get ray direction
    var uv = (fragCoord - 0.5 * rez) / rez.y;
    uv.y = -uv.y;
    var rd = camera * normalize(vec3(uv, 1.0));

    // call march function and get the color/depth
    var march_result = march(ro, rd);

    // Get current position for lighting calculation
    var current = ro + rd * march_result.depth;

    // If we hit nothing (depth >= MAX_DIST), show background
    var color = march_result.color;
    if (march_result.depth >= MAX_DIST) {
        var light_position = vec3f(uniforms[13], uniforms[14], uniforms[15]);
        var sun_color = int_to_rgb(i32(uniforms[16]));
        color = get_ambient_light(light_position, sun_color, rd);
    } else {
        // We hit something, calculate lighting
        color = get_light(current, march_result.color, rd);
        // color = march_result.color;
    }
    
    // display the result
    color = linear_to_gamma(color);
    fb[mapfb(id.xy, uniforms[1])] = vec4f(color, 1.0);
}