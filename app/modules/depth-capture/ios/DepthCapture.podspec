require 'json'

package = JSON.parse(File.read(File.join(__dir__, '..', '..', '..', 'package.json')))

Pod::Spec.new do |s|
  s.name           = 'DepthCapture'
  s.version        = package['version']
  s.summary        = 'Experimental one-shot LiDAR depth capture for plate-vision.'
  s.description    = 'Returns a single depth map as unsigned millimetres, unfiltered, for ' \
                     'comparison against the Nutrition5k depth distribution the model was ' \
                     'trained on. Diagnostic only: it does not change any estimate.'
  s.author         = 'Simon Kundrik'
  s.homepage       = 'https://github.com/simonkundrik/plate-vision'
  s.license        = { :type => 'MIT' }
  s.platforms      = { :ios => '15.1' }
  s.source         = { git: 'https://github.com/simonkundrik/plate-vision.git' }
  s.static_framework = true

  s.dependency 'ExpoModulesCore'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'SWIFT_COMPILATION_MODE' => 'wholemodule'
  }

  s.source_files = "**/*.{h,m,mm,swift,hpp,cpp}"
end
