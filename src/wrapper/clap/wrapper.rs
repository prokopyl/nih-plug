use atomic_float::AtomicF32;
use atomic_refcell::{AtomicRefCell, AtomicRefMut};
use clack_extensions::audio_ports::{
    AudioPortFlags, AudioPortInfoData, AudioPortInfoWriter, AudioPortType, PluginAudioPorts,
    PluginAudioPortsImpl,
};
use clack_extensions::audio_ports_config::{
    AudioPortConfigSelectError, AudioPortConfigWriter, AudioPortsConfiguration, MainPortInfo,
    PluginAudioPortsConfig, PluginAudioPortsConfigImpl,
};
use clack_extensions::gui::{
    GuiApiType, GuiError, GuiResizeHints, GuiSize, HostGui, PluginGui, PluginGuiImpl, Window,
};
use clack_extensions::latency::{HostLatency, PluginLatency, PluginLatencyImpl};
use clack_extensions::note_ports::{
    NoteDialect, NoteDialects, NotePortInfoData, NotePortInfoWriter, PluginNotePorts,
    PluginNotePortsImpl,
use clap_sys::ext::draft::remote_controls::{
    clap_plugin_remote_controls, clap_remote_controls_page, CLAP_EXT_REMOTE_CONTROLS,
};
use clack_extensions::params::implementation::{
    ParamDisplayWriter, ParamInfoWriter, PluginMainThreadParams, PluginParamsImpl,
};
use clack_extensions::params::info::{ParamInfoData, ParamInfoFlags};
use clack_extensions::params::{HostParams, ParamRescanFlags, PluginParams};
use clack_extensions::render::{PluginRender, PluginRenderError, PluginRenderImpl, RenderMode};
use clack_extensions::state::PluginStateImpl;
use clack_extensions::tail::{PluginTail, PluginTailImpl, TailLength};
use clack_extensions::thread_check::HostThreadCheck;
use clack_extensions::voice_info::{
    HostVoiceInfo, PluginVoiceInfo, PluginVoiceInfoImpl, VoiceInfo, VoiceInfoFlags,
};
use clack_plugin::events::event_types::{
    MidiEvent, MidiSysExEvent, NoteChokeEvent, NoteEndEvent, NoteExpressionEvent,
    NoteExpressionType, NoteOffEvent, NoteOnEvent, ParamGestureBeginEvent, ParamGestureEndEvent,
    ParamValueEvent, TransportEvent, TransportFlags,
};
use clack_plugin::events::spaces::CoreEventSpace;
use clack_plugin::events::{EventFlags, EventHeader};
use clack_plugin::extensions::PluginExtensions;
use clack_plugin::host::{HostAudioThreadHandle, HostHandle, HostMainThreadHandle};
use clack_plugin::plugin::{AudioConfiguration, PluginError};
use clack_plugin::prelude::{
    Audio, Events, InputEvents, OutputEvents, PluginMainThread, PluginShared, Process, UnknownEvent,
};
use clack_plugin::process::audio::ChannelPair;
use clack_plugin::stream::{InputStream, OutputStream};
use clack_plugin::utils::Cookie;
use crossbeam::atomic::AtomicCell;
use crossbeam::channel::{self, SendTimeoutError};
use crossbeam::queue::ArrayQueue;
use parking_lot::Mutex;
use raw_window_handle::HasRawWindowHandle;
use std::any::Any;
use std::borrow::Borrow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::{self, ThreadId};
use std::time::Duration;

use super::context::{WrapperGuiContext, WrapperInitContext, WrapperProcessContext};
use super::descriptor::PluginDescriptor;
use crate::buffer::Buffer;
use crate::context::gui::AsyncExecutor;
use crate::context::process::Transport;
use crate::editor::{Editor, ParentWindowHandle};
use crate::event_loop::{BackgroundThread, EventLoop, MainThreadExecutor, TASK_QUEUE_CAPACITY};
use crate::midi::MidiResult;
use crate::prelude::{
    AsyncExecutor, AudioIOLayout, AuxiliaryBuffers, BufferConfig, ClapPlugin, Editor, MidiConfig,
    NoteEvent, ParamFlags, ParamPtr, Params, ParentWindowHandle, Plugin, PluginNoteEvent,
    ProcessMode, ProcessStatus, SysExMessage, TaskExecutor, Transport,
};
use crate::util::permit_alloc;
use crate::wrapper::state::{self, PluginState};
use crate::wrapper::util::buffer_management::{BufferManager, ChannelPointers};
use crate::wrapper::util::{
    clamp_input_event_timing, clamp_output_event_timing, hash_param_id, process_wrapper,
};

/// How many output parameter changes we can store in our output parameter change queue. Storing
/// more than this many parameters at a time will cause changes to get lost.
const OUTPUT_EVENT_QUEUE_CAPACITY: usize = 2048;

pub struct ClapWrapperAudioProcessor<'a, P: ClapPlugin> {
    shared: Arc<Wrapper<'a, P>>,
    host: HostAudioThreadHandle<'a>,
}

pub struct ClapWrapperMainThread<'a, P: ClapPlugin> {
    shared: Arc<Wrapper<'a, P>>,
    host: HostMainThreadHandle<'a>,
}

pub struct ClapWrapperShared<'a, P: ClapPlugin> {
    wrapper: Arc<Wrapper<'a, P>>,
}

pub struct Wrapper<'a, P: ClapPlugin> {
    /// The wrapped plugin instance.
    plugin: Mutex<P>,
    /// The plugin's background task executor closure.
    pub task_executor: Mutex<TaskExecutor<P>>,
    /// The plugin's parameters. These are fetched once during initialization. That way the
    /// `ParamPtr`s are guaranteed to live at least as long as this object and we can interact with
    /// the `Params` object without having to acquire a lock on `plugin`.
    params: Arc<dyn Params>,
    /// The plugin's editor, if it has one. This object does not do anything on its own, but we need
    /// to instantiate this in advance so we don't need to lock the entire [`Plugin`] object when
    /// creating an editor. Wrapped in an `AtomicRefCell` because it needs to be initialized late.
    editor: AtomicRefCell<Option<Mutex<Box<dyn Editor>>>>,
    /// A handle for the currently active editor instance. The plugin should implement `Drop` on
    /// this handle for its closing behavior.
    editor_handle: Mutex<Option<Box<dyn Any + Send>>>,
    /// The DPI scaling factor as passed to the [IPlugViewContentScaleSupport::set_scale_factor()]
    /// function. Defaults to 1.0, and will be kept there on macOS. When reporting and handling size
    /// the sizes communicated to and from the DAW should be scaled by this factor since NIH-plug's
    /// APIs only deal in logical pixels.
    editor_scaling_factor: AtomicF32,

    is_processing: AtomicBool,
    /// The current IO configuration, modified through the `clap_plugin_audio_ports_config`
    /// extension. Initialized to the plugin's first audio IO configuration.
    current_audio_io_layout: AtomicCell<AudioIOLayout>,
    /// The current buffer configuration, containing the sample rate and the maximum block size.
    /// Will be set in `clap_plugin::activate()`.
    current_buffer_config: AtomicCell<Option<BufferConfig>>,
    /// The current audio processing mode. Set through the render extension. Defaults to realtime.
    pub current_process_mode: AtomicCell<ProcessMode>,
    /// The incoming events for the plugin, if `P::MIDI_INPUT` is set to `MidiConfig::Basic` or
    /// higher.
    ///
    /// TODO: Maybe load these lazily at some point instead of needing to spool them all to this
    ///       queue first
    input_events: AtomicRefCell<VecDeque<PluginNoteEvent<P>>>,
    /// Stores any events the plugin has output during the current processing cycle, analogous to
    /// `input_events`.
    output_events: AtomicRefCell<VecDeque<PluginNoteEvent<P>>>,
    /// The last process status returned by the plugin. This is used for tail handling.
    last_process_status: AtomicCell<ProcessStatus>,
    /// The current latency in samples, as set by the plugin through the [`ProcessContext`]. Uses
    /// the latency extension.
    pub current_latency: AtomicU32,
    /// A data structure that helps manage and create buffers for all of the plugin's inputs and
    /// outputs based on channel pointers provided by the host.
    buffer_manager: AtomicRefCell<BufferManager>,
    /// The plugin is able to restore state through a method on the `GuiContext`. To avoid changing
    /// parameters mid-processing and running into garbled data if the host also tries to load state
    /// at the same time the restoring happens at the end of each processing call. If this zero
    /// capacity channel contains state data at that point, then the audio thread will take the
    /// state out of the channel, restore the state, and then send it back through the same channel.
    /// In other words, the GUI thread acts as a sender and then as a receiver, while the audio
    /// thread acts as a receiver and then as a sender. That way deallocation can happen on the GUI
    /// thread. All of this happens without any blocking on the audio thread.
    updated_state_sender: channel::Sender<PluginState>,
    /// The receiver belonging to [`new_state_sender`][Self::new_state_sender].
    updated_state_receiver: channel::Receiver<PluginState>,

    // We'll query all of the host's extensions upfront
    host: HostHandle<'a>,

    clap_plugin_audio_ports_config: clap_plugin_audio_ports_config,

    /// Needs to be boxed because the plugin object is supposed to contain a static reference to
    /// this.
    _plugin_descriptor: Box<PluginDescriptor<P>>,

    host_gui: Option<&'a HostGui>,

    host_latency: Option<&'a HostLatency>,

    host_params: Option<&'a HostParams>,
    // These fields are exactly the same as their VST3 wrapper counterparts.
    //
    /// The keys from `param_map` in a stable order.
    param_hashes: Vec<u32>,
    // TODO: Merge the three `*_by_hash` hashmaps at some point
    /// A mapping from parameter ID hashes (obtained from the string parameter IDs) to pointers to
    /// parameters belonging to the plugin. These addresses will remain stable as long as the
    /// `params` object does not get deallocated.
    param_by_hash: HashMap<u32, ParamPtr>,
    /// Mappings from parameter hashes to string parameter IDs. Used for notifying the plugin's
    /// editor about parameter changes.
    param_id_by_hash: HashMap<u32, String>,
    /// The group name of a parameter, indexed by the parameter's hash. Nested groups are delimited
    /// by slashes, and they're only used to allow the DAW to display parameters in a tree
    /// structure.
    param_group_by_hash: HashMap<u32, String>,
    /// Mappings from string parameter identifiers to parameter hashes. Useful for debug logging
    /// and when storing and restoring plugin state.
    param_id_to_hash: HashMap<String, u32>,
    /// The inverse mapping from [`param_by_hash`][Self::param_by_hash]. This is needed to be able
    /// to have an ergonomic parameter setting API that uses references to the parameters instead of
    /// having to add a setter function to the parameter (or even worse, have it be completely
    /// untyped).
    pub param_ptr_to_hash: HashMap<ParamPtr, u32>,
    /// For all polyphonically modulatable parameters, mappings from the parameter hash's hash to
    /// the parameter's poly modulation ID. These IDs are then passed to the plugin, so it can
    /// quickly refer to parameter by matching on constant IDs.
    poly_mod_ids_by_hash: HashMap<u32, u32>,
    /// A queue of parameter changes and gestures that should be output in either the next process
    /// call or in the next parameter flush.
    ///
    /// XXX: There's no guarantee that a single parameter doesn't occur twice in this queue, but
    ///      even if it does then that should still not be a problem because the host also reads it
    ///      in the same order, right?
    output_parameter_events: ArrayQueue<OutputParamEvent>,

    host_thread_check: Option<&'a HostThreadCheck>,

    clap_plugin_remote_controls: clap_plugin_remote_controls,
    /// The plugin's remote control pages, if it defines any. Filled when initializing the plugin.
    remote_control_pages: Vec<clap_remote_controls_page>,

    clap_plugin_render: clap_plugin_render,

    // TODO: support voice info
    host_voice_info: Option<&'a HostVoiceInfo>,
    // host_voice_info: AtomicRefCell<Option<ClapPtr<clap_host_voice_info>>>,
    /// If `P::CLAP_POLY_MODULATION_CONFIG` is set, then the plugin can configure the current number
    /// of active voices using a context method called from the initialization or processing
    /// context. This defaults to the maximum number of voices.
    current_voice_capacity: AtomicU32,

    /// A queue of tasks that still need to be performed. Because CLAP lets the plugin request a
    /// host callback directly, we don't need to use the OsEventLoop we use in our other plugin
    /// implementations. Instead, we'll post tasks to this queue, ask the host to call
    /// [`on_main_thread()`][Self::on_main_thread()] on the main thread, and then continue to pop
    /// tasks off this queue there until it is empty.
    tasks: ArrayQueue<Task<P>>,
    /// The ID of the main thread. In practice this is the ID of the thread that created this
    /// object. If the host supports the thread check extension (and
    /// [`host_thread_check`][Self::host_thread_check] thus contains a value), then that extension
    /// is used instead.
    main_thread_id: ThreadId,
    /// A background thread for running tasks independently from the host'main GUI thread. Useful
    /// for longer, blocking tasks. Initialized later as it needs a reference to the wrapper.
    background_thread: AtomicRefCell<Option<BackgroundThread<Task<P>, Wrapper<'a, P>>>>,
}

/// Tasks that can be sent from the plugin to be executed on the main thread in a non-blocking
/// realtime-safe way. Instead of using a random thread or the OS' event loop like in the Linux
/// implementation, this uses [`clap_host::request_callback()`] instead.
#[allow(clippy::enum_variant_names)]
pub enum Task<P: Plugin> {
    /// Execute one of the plugin's background tasks.
    PluginTask(P::BackgroundTask),
    /// Inform the plugin that one or more parameter values have changed.
    ParameterValuesChanged,
    /// Inform the plugin that one parameter's value has changed. This uses the parameter hashes
    /// since the task will be created from the audio thread.
    ParameterValueChanged(u32, f32),
    /// Inform the plugin that one parameter's modulation offset has changed. This uses the
    /// parameter hashes since the task will be created from the audio thread.
    ParameterModulationChanged(u32, f32),
    /// Inform the host that the latency has changed.
    LatencyChanged,
    /// Inform the host that the voice info has changed.
    VoiceInfoChanged,
    /// Tell the host that it should rescan the current parameter values.
    RescanParamValues,
}

/// The types of CLAP parameter updates for events.
pub enum ClapParamUpdate {
    /// Set the parameter to this plain value. In our wrapper the plain values are the normalized
    /// values multiplied by the step count for discrete parameters.
    PlainValueSet(f64),
    /// Set a normalized offset for the parameter's plain value. Subsequent modulation events
    /// override the previous one, but `PlainValueSet`s do not override the existing modulation.
    /// These values should also be divided by the step size.
    PlainValueMod(f64),
}

/// A parameter event that should be output by the plugin, stored in a queue on the wrapper and
/// written to the host either at the end of the process function or during a flush.
#[derive(Debug, Clone)]
pub enum OutputParamEvent {
    /// Begin an automation gesture. This must always be sent before sending [`SetValue`].
    BeginGesture { param_hash: u32 },
    /// Change the value of a parameter using a plain CLAP value, aka the normalized value
    /// multiplied by the number of steps.
    SetValue {
        /// The internal hash for the parameter.
        param_hash: u32,
        /// The 'plain' value as reported to CLAP. This is the normalized value multiplied by
        /// [`params::step_size()`][crate::params::step_size()].
        clap_plain_value: f64,
    },
    /// Begin an automation gesture. This must always be sent after sending one or more [`SetValue`]
    /// events.
    EndGesture { param_hash: u32 },
}

/// Because CLAP has this [`clap_host::request_host_callback()`] function, we don't need to use
/// `OsEventLoop` and can instead just request a main thread callback directly.
impl<P: ClapPlugin> EventLoop<Task<P>, Wrapper<P>> for Wrapper<P> {
    fn new_and_spawn(_executor: Weak<Self>) -> Self {
        panic!("What are you doing");
    }

    fn schedule_gui(&self, task: Task<P>) -> bool {
        if self.is_main_thread() {
            self.execute(task, true);
            true
        } else {
            let success = self.tasks.push(task).is_ok();
            if success {
                // CLAP lets us use the host's event loop instead of having to implement our own
                self.host.request_callback();
            }

            success
        }
    }

    fn schedule_background(&self, task: Task<P>) -> bool {
        self.background_thread
            .borrow()
            .as_ref()
            .unwrap()
            .schedule(task)
    }

    fn is_main_thread(&self) -> bool {
        // If the host supports the thread check interface then we'll use that, otherwise we'll
        // check if this is the same thread as the one that created the plugin instance.
        self.host_thread_check
            .and_then(|thread_check| thread_check.is_main_thread(&self.host))
            // FIXME: `thread::current()` may allocate the first time it's called, is there a safe
            //        non-allocating version of this without using huge OS-specific libraries?
            .unwrap_or_else(|| permit_alloc(|| thread::current().id() == self.main_thread_id))
    }
}

impl<'a, P: ClapPlugin> MainThreadExecutor<Task<P>> for Wrapper<'a, P> {
    fn execute(&self, task: Task<P>, is_gui_thread: bool) {
        // This function is always called from the main thread, from [Self::on_main_thread].
        match task {
            Task::PluginTask(task) => (self.task_executor.lock())(task),
            Task::ParameterValuesChanged => {
                if self.editor_handle.lock().is_some() {
                    if let Some(editor) = self.editor.borrow().as_ref() {
                        editor.lock().param_values_changed();
                    }
                }
            }
            Task::ParameterValueChanged(param_hash, normalized_value) => {
                if self.editor_handle.lock().is_some() {
                    if let Some(editor) = self.editor.borrow().as_ref() {
                        let param_id = &self.param_id_by_hash[&param_hash];
                        editor
                            .lock()
                            .param_value_changed(param_id, normalized_value);
                    }
                }
            }
            Task::ParameterModulationChanged(param_hash, modulation_offset) => {
                if self.editor_handle.lock().is_some() {
                    if let Some(editor) = self.editor.borrow().as_ref() {
                        let param_id = &self.param_id_by_hash[&param_hash];
                        editor
                            .lock()
                            .param_modulation_changed(param_id, modulation_offset);
                    }
                }
            }
            Task::LatencyChanged => match &self.host_latency {
                Some(host_latency) => {
                    nih_debug_assert!(is_gui_thread);

                    // XXX: The CLAP docs mention that you should request a restart if this happens
                    //      while the plugin is activated (which is not entirely the same thing as
                    //      is processing, but we'll treat it as the same thing). In practice just
                    //      calling the latency changed function also seems to work just fine.
                    if self.is_processing.load(Ordering::SeqCst) {
                        self.host.request_restart();
                    } else {
                        // FIXME: This is unsound: it should only be called on the Main Thread with exclusive access.
                        unsafe {
                            host_latency.changed(&mut self.host.as_main_thread_unchecked());
                        }
                    }
                }
                None => nih_debug_assert_failure!("Host does not support the latency extension"),
            },
            Task::VoiceInfoChanged => match &*self.host_voice_info.borrow() {
                Some(host_voice_info) => {
                    nih_debug_assert!(is_gui_thread);
                    // FIXME: This is unsound: it should only be called on the Main Thread with exclusive access.
                    unsafe {
                        host_voice_info.changed(&mut self.host.as_main_thread_unchecked());
                    }
                }
                None => nih_debug_assert_failure!("Host does not support the voice-info extension"),
            },
            Task::RescanParamValues => match &*self.host_params.borrow() {
                Some(host_params) => {
                    nih_debug_assert!(is_gui_thread);
                    // FIXME: This is unsound: it should only be called on the Main Thread with exclusive access.
                    unsafe {
                        host_params.rescan(
                            &mut self.host.as_main_thread_unchecked(),
                            ParamRescanFlags::VALUES,
                        );
                    }
                }
                None => nih_debug_assert_failure!("The host does not support parameters? What?"),
            },
        };
    }
}

impl<'a, P: ClapPlugin> PluginShared<'a> for ClapWrapperShared<'a, P> {
    fn new(host: HostHandle<'a>) -> Result<Self, PluginError> {
        let mut plugin = P::default();
        let task_executor = Mutex::new(plugin.task_executor());

        // This is used to allow the plugin to restore preset data from its editor, see the comment
        // on `Self::updated_state_sender`
        let (updated_state_sender, updated_state_receiver) = channel::bounded(0);

        let plugin_descriptor: Box<PluginDescriptor<P>> = Box::default();

        // This is a mapping from the parameter IDs specified by the plugin to pointers to those
        // parameters. These pointers are assumed to be safe to dereference as long as
        // `wrapper.plugin` is alive. The plugin API identifiers these parameters by hashes, which
        // we'll calculate from the string ID specified by the plugin. These parameters should also
        // remain in the same order as the one returned by the plugin.
        let params = plugin.params();
        let param_id_hashes_ptrs_groups: Vec<_> = params
            .param_map()
            .into_iter()
            .map(|(id, ptr, group)| {
                let hash = hash_param_id(&id);
                (id, hash, ptr, group)
            })
            .collect();
        let param_hashes = param_id_hashes_ptrs_groups
            .iter()
            .map(|(_, hash, _, _)| *hash)
            .collect();
        let param_by_hash = param_id_hashes_ptrs_groups
            .iter()
            .map(|(_, hash, ptr, _)| (*hash, *ptr))
            .collect();
        let param_id_by_hash = param_id_hashes_ptrs_groups
            .iter()
            .map(|(id, hash, _, _)| (*hash, id.clone()))
            .collect();
        let param_group_by_hash = param_id_hashes_ptrs_groups
            .iter()
            .map(|(_, hash, _, group)| (*hash, group.clone()))
            .collect();
        let param_id_to_hash = param_id_hashes_ptrs_groups
            .iter()
            .map(|(id, hash, _, _)| (id.clone(), *hash))
            .collect();
        let param_ptr_to_hash = param_id_hashes_ptrs_groups
            .iter()
            .map(|(_, hash, ptr, _)| (*ptr, *hash))
            .collect();
        let poly_mod_ids_by_hash: HashMap<u32, u32> = param_id_hashes_ptrs_groups
            .iter()
            .filter_map(|(_, hash, ptr, _)| unsafe {
                ptr.poly_modulation_id().map(|id| (*hash, id))
            })
            .collect();

        if cfg!(debug_assertions) {
            let param_map = params.param_map();
            let param_ids: HashSet<_> = param_id_hashes_ptrs_groups
                .iter()
                .map(|(id, _, _, _)| id.clone())
                .collect();
            nih_debug_assert_eq!(
                param_map.len(),
                param_ids.len(),
                "The plugin has duplicate parameter IDs, weird things may happen. Consider using \
                 6 character parameter IDs to avoid collisions."
            );

            let poly_mod_ids: HashSet<u32> = poly_mod_ids_by_hash.values().copied().collect();
            nih_debug_assert_eq!(
                poly_mod_ids_by_hash.len(),
                poly_mod_ids.len(),
                "The plugin has duplicate poly modulation IDs. Polyphonic modulation will not be \
                 routed to the correct parameter."
            );

            let mut bypass_param_exists = false;
            for (_, _, ptr, _) in &param_id_hashes_ptrs_groups {
                let flags = unsafe { ptr.flags() };
                let is_bypass = flags.contains(ParamFlags::BYPASS);

                if is_bypass && bypass_param_exists {
                    nih_debug_assert_failure!(
                        "Duplicate bypass parameters found, the host will only use the first one"
                    );
                }

                bypass_param_exists |= is_bypass;
            }
        }

        // Support for the remote controls extension
        let mut remote_control_pages = Vec::new();
        RemoteControlPages::define_remote_control_pages(
            &plugin,
            &mut remote_control_pages,
            &param_ptr_to_hash,
        );

        let wrapper = Arc::new(Wrapper {
            plugin: Mutex::new(plugin),
            task_executor,
            params,
            // Initialized later as it needs a reference to the wrapper for the async executor
            editor: AtomicRefCell::new(None),
            editor_handle: Mutex::new(None),
            editor_scaling_factor: AtomicF32::new(1.0),

            is_processing: AtomicBool::new(false),
            current_audio_io_layout: AtomicCell::new(
                P::AUDIO_IO_LAYOUTS.first().copied().unwrap_or_default(),
            ),
            current_buffer_config: AtomicCell::new(None),
            current_process_mode: AtomicCell::new(ProcessMode::Realtime),
            input_events: AtomicRefCell::new(VecDeque::with_capacity(512)),
            output_events: AtomicRefCell::new(VecDeque::with_capacity(512)),
            last_process_status: AtomicCell::new(ProcessStatus::Normal),
            current_latency: AtomicU32::new(0),
            // This is initialized just before calling `Plugin::initialize()` so that during the
            // process call buffers can be initialized without any allocations
            buffer_manager: AtomicRefCell::new(BufferManager::for_audio_io_layout(
                0,
                AudioIOLayout::default(),
            )),
            updated_state_sender,
            updated_state_receiver,

            host,

            _plugin_descriptor: plugin_descriptor,

            supported_bus_configs,

            host_gui: host.extension(),

            host_latency: host.extension(),

            host_params: host.extension(),
            param_hashes,
            param_by_hash,
            param_id_by_hash,
            param_group_by_hash,
            param_id_to_hash,
            param_ptr_to_hash,
            poly_mod_ids_by_hash,
            output_parameter_events: ArrayQueue::new(OUTPUT_EVENT_QUEUE_CAPACITY),

            host_thread_check: host.extension(),
            remote_control_pages,

            host_voice_info: host.extension(),
            current_voice_capacity: AtomicU32::new(
                P::CLAP_POLY_MODULATION_CONFIG
                    .map(|c| {
                        nih_debug_assert!(
                            c.max_voice_capacity >= 1,
                            "The maximum voice capacity cannot be zero"
                        );
                        c.max_voice_capacity
                    })
                    .unwrap_or(1),
            ),

            tasks: ArrayQueue::new(TASK_QUEUE_CAPACITY),
            main_thread_id: thread::current().id(),
            // Initialized later as it needs a reference to the wrapper for the executor
            background_thread: AtomicRefCell::new(None),
        });

        // The editor also needs to be initialized later so the Async executor can work.
        *wrapper.editor.borrow_mut() = wrapper
            .plugin
            .lock()
            .editor(AsyncExecutor {
                execute_background: Arc::new({
                    let wrapper = wrapper.clone();

                    move |task| {
                        let task_posted = wrapper.schedule_background(Task::PluginTask(task));
                        nih_debug_assert!(task_posted, "The task queue is full, dropping task...");
                    }
                }),
                execute_gui: Arc::new({
                    let wrapper = wrapper.clone();

                    move |task| {
                        let task_posted = wrapper.schedule_gui(Task::PluginTask(task));
                        nih_debug_assert!(task_posted, "The task queue is full, dropping task...");
                    }
                }),
            })
            .map(Mutex::new);

        // Same with the background thread
        *wrapper.background_thread.borrow_mut() =
            Some(BackgroundThread::get_or_create(Arc::downgrade(&wrapper)));

        Ok(Self { wrapper })
    }
}

impl<'a, P: ClapPlugin> Wrapper<'a, P> {
    fn make_gui_context(self: Arc<Self>) -> Arc<WrapperGuiContext<'a, P>> {
        Arc::new(WrapperGuiContext {
            wrapper: self,
            #[cfg(debug_assertions)]
            param_gesture_checker: Default::default(),
        })
    }

    /// # Note
    ///
    /// The lock on the plugin must be dropped before this object is dropped to avoid deadlocks
    /// caused by reentrant function calls.
    fn make_init_context(&self) -> WrapperInitContext<'_, P> {
        WrapperInitContext {
            wrapper: self,
            pending_requests: Default::default(),
        }
    }

    fn make_process_context(&self, transport: Transport) -> WrapperProcessContext<'_, P> {
        WrapperProcessContext {
            wrapper: self,
            input_events_guard: self.input_events.borrow_mut(),
            output_events_guard: self.output_events.borrow_mut(),
            transport,
        }
    }

    /// Get a parameter's ID based on a `ParamPtr`. Used in the `GuiContext` implementation for the
    /// gesture checks.
    #[allow(unused)]
    pub fn param_id_from_ptr(&self, param: ParamPtr) -> Option<&str> {
        self.param_ptr_to_hash
            .get(&param)
            .and_then(|hash| self.param_id_by_hash.get(hash))
            .map(|s| s.as_str())
    }

    /// Queue a parameter output event to be sent to the host at the end of the audio processing
    /// cycle, and request a parameter flush from the host if the plugin is not currently processing
    /// audio. The parameter's actual value will only be updated at that point so the value won't
    /// change in the middle of a processing call.
    ///
    /// Returns `false` if the parameter value queue was full and the update will not be sent to the
    /// host (it will still be set on the plugin either way).
    pub fn queue_parameter_event(&self, event: OutputParamEvent) -> bool {
        let result = self.output_parameter_events.push(event).is_ok();

        // Requesting a flush is fine even during audio processing. This avoids a race condition.
        match self.host_params {
            Some(host_params) => host_params.request_flush(&self.host),
            None => nih_debug_assert_failure!("The host does not support parameters? What?"),
        }

        result
    }

    /// Request a resize based on the editor's current reported size. As of CLAP 0.24 this can
    /// safely be called from any thread. If this returns `false`, then the plugin should reset its
    /// size back to the previous value.
    pub fn request_resize(&self) -> bool {
        match (self.host_gui, self.editor.borrow().as_ref()) {
            (Some(host_gui), Some(editor)) => {
                let (unscaled_width, unscaled_height) = editor.lock().size();
                let scaling_factor = self.editor_scaling_factor.load(Ordering::Relaxed);

                host_gui
                    .request_resize(
                        &self.host,
                        (unscaled_width as f32 * scaling_factor).round() as u32,
                        (unscaled_height as f32 * scaling_factor).round() as u32,
                    )
                    .is_ok()
            }
            _ => false,
        }
    }

    /// Convenience function for setting a value for a parameter as triggered by a VST3 parameter
    /// update. The same rate is for updating parameter smoothing.
    ///
    /// After calling this function, you should call
    /// [`notify_param_values_changed()`][Self::notify_param_values_changed()] to allow the editor
    /// to update itself. This needs to be done separately so you can process parameter changes in
    /// batches.
    ///
    /// # Note
    ///
    /// These values are CLAP plain values, which include a step count multiplier for discrete
    /// parameter values.
    pub fn update_plain_value_by_hash(
        &self,
        hash: u32,
        update_type: ClapParamUpdate,
        sample_rate: Option<f32>,
    ) -> bool {
        match self.param_by_hash.get(&hash) {
            Some(param_ptr) => {
                match update_type {
                    ClapParamUpdate::PlainValueSet(clap_plain_value) => {
                        let normalized_value = clap_plain_value as f32
                            / unsafe { param_ptr.step_count() }.unwrap_or(1) as f32;

                        if unsafe { param_ptr.set_normalized_value(normalized_value) } {
                            if let Some(sample_rate) = sample_rate {
                                unsafe { param_ptr.update_smoother(sample_rate, false) };
                            }

                            // The GUI needs to be informed about the changed parameter value. This
                            // triggers an `Editor::param_value_changed()` call on the GUI thread.
                            let task_posted = self
                                .schedule_gui(Task::ParameterValueChanged(hash, normalized_value));
                            nih_debug_assert!(
                                task_posted,
                                "The task queue is full, dropping task..."
                            );
                        }

                        true
                    }
                    ClapParamUpdate::PlainValueMod(clap_plain_delta) => {
                        let normalized_delta = clap_plain_delta as f32
                            / unsafe { param_ptr.step_count() }.unwrap_or(1) as f32;

                        if unsafe { param_ptr.modulate_value(normalized_delta) } {
                            if let Some(sample_rate) = sample_rate {
                                unsafe { param_ptr.update_smoother(sample_rate, false) };
                            }

                            let task_posted = self.schedule_gui(Task::ParameterModulationChanged(
                                hash,
                                normalized_delta,
                            ));
                            nih_debug_assert!(
                                task_posted,
                                "The task queue is full, dropping task..."
                            );
                        }

                        true
                    }
                }
            }
            _ => false,
        }
    }

    /// Handle all incoming events from an event queue. This will clear `self.input_events` first.
    pub fn handle_in_events(
        &self,
        in_: &InputEvents,
        current_sample_idx: usize,
        total_buffer_len: usize,
    ) {
        let mut input_events = self.input_events.borrow_mut();
        input_events.clear();

        for event in in_ {
            self.handle_in_event(
                event,
                &mut input_events,
                None,
                current_sample_idx,
                total_buffer_len,
            );
        }
    }

    /// Similar to [`handle_in_events()`][Self::handle_in_events()], but will stop just before an
    /// event if the predicate returns true for that events. This predicate is only called for
    /// events that occur after `current_sample_idx`. This is used to stop before a tempo or time
    /// signature change, or before next parameter change event with `raw_event.time >
    /// current_sample_idx` and return the **absolute** (relative to the entire buffer that's being
    /// split) sample index of that event along with the its index in the event queue as a
    /// `(sample_idx, event_idx)` tuple. This allows for splitting the audio buffer into segments
    /// with distinct sample values to enable sample accurate automation without modifications to the
    /// wrapped plugin.
    pub fn handle_in_events_until(
        &self,
        in_: &InputEvents,
        transport_info: &mut &TransportEvent,
        current_sample_idx: usize,
        total_buffer_len: usize,
        resume_from_event_idx: usize,
        stop_predicate: impl Fn(&UnknownEvent) -> bool,
    ) -> Option<(usize, usize)> {
        let mut input_events = self.input_events.borrow_mut();
        input_events.clear();

        // To achieve this, we'll always read one event ahead
        let num_events = in_.len();
        if num_events == 0 {
            return None;
        }

        let start_idx = resume_from_event_idx as u32;
        let mut event = in_.get(start_idx)?;

        for next_event_idx in (start_idx + 1)..num_events {
            self.handle_in_event(
                event,
                &mut input_events,
                Some(transport_info),
                current_sample_idx,
                total_buffer_len,
            );

            // Stop just before the next parameter change or transport information event at a sample
            // after the current sample
            event = in_.get(next_event_idx)?;
            if event.header().time() > current_sample_idx as u32 && stop_predicate(event) {
                return Some((event.header().time() as usize, next_event_idx as usize));
            }
        }

        // Don't forget about the last event
        self.handle_in_event(
            event,
            &mut input_events,
            Some(transport_info),
            current_sample_idx,
            total_buffer_len,
        );

        None
    }

    /// Write the unflushed parameter changes to the host's output event queue. The sample index is
    /// used as part of splitting up the input buffer for sample accurate automation changes. This
    /// will also modify the actual parameter values, since we should only do that while the wrapped
    /// plugin is not actually processing audio.
    ///
    /// The `total_buffer_len` argument is used to clamp out of bounds events to the buffer's length.
    pub fn handle_out_events(
        &self,
        out: &mut OutputEvents,
        current_sample_idx: usize,
        total_buffer_len: usize,
    ) {
        // We'll always write these events to the first sample, so even when we add note output we
        // shouldn't have to think about interleaving events here
        let sample_rate = self.current_buffer_config.load().map(|c| c.sample_rate);
        while let Some(change) = self.output_parameter_events.pop() {
            let push_successful = match change {
                OutputParamEvent::BeginGesture { param_hash } => {
                    out.try_push(ParamGestureBeginEvent::new(
                        EventHeader::new_core(current_sample_idx as u32, EventFlags::IS_LIVE),
                        param_hash,
                    ))
                }
                OutputParamEvent::SetValue {
                    param_hash,
                    clap_plain_value,
                } => {
                    self.update_plain_value_by_hash(
                        param_hash,
                        ClapParamUpdate::PlainValueSet(clap_plain_value),
                        sample_rate,
                    );

                    out.try_push(ParamValueEvent::new(
                        EventHeader::new_core(current_sample_idx as u32, EventFlags::IS_LIVE),
                        Cookie::empty(),
                        -1,
                        param_hash,
                        -1,
                        -1,
                        -1,
                        clap_plain_value,
                    ))
                }
                OutputParamEvent::EndGesture { param_hash } => {
                    out.try_push(ParamGestureEndEvent::new(
                        EventHeader::new_core(current_sample_idx as u32, EventFlags::IS_LIVE),
                        param_hash,
                    ))
                }
            };

            nih_debug_assert!(push_successful.is_ok());
        }

        // Also send all note events generated by the plugin
        let mut output_events = self.output_events.borrow_mut();
        while let Some(event) = output_events.pop_front() {
            // Out of bounds events are clamped to the buffer's size
            let time = clamp_output_event_timing(
                event.timing() + current_sample_idx as u32,
                total_buffer_len as u32,
            );

            let push_successful = match event {
                NoteEvent::NoteOn {
                    timing: _,
                    voice_id,
                    channel,
                    note,
                    velocity,
                } if P::MIDI_OUTPUT >= MidiConfig::Basic => {
                    out.try_push(NoteOnEvent(
                        clack_plugin::events::event_types::NoteEvent::new(
                            EventHeader::new_core(
                                time,
                                // We don't have a way to denote live events
                                EventFlags::empty(),
                            ),
                            voice_id.unwrap_or(-1),
                            0,
                            note as i16,
                            channel as i16,
                            velocity as f64,
                        ),
                    ))
                }
                NoteEvent::NoteOff {
                    timing: _,
                    voice_id,
                    channel,
                    note,
                    velocity,
                } if P::MIDI_OUTPUT >= MidiConfig::Basic => out.try_push(NoteOffEvent(
                    clack_plugin::events::event_types::NoteEvent::new(
                        EventHeader::new_core(time, EventFlags::empty()),
                        voice_id.unwrap_or(-1),
                        0,
                        note as i16,
                        channel as i16,
                        velocity as f64,
                    ),
                )),
                // NOTE: This is gated behind `P::MIDI_INPUT`, because this is a merely a hint event
                //       for the host. It is not output to any other plugin or device.
                NoteEvent::VoiceTerminated {
                    timing: _,
                    voice_id,
                    channel,
                    note,
                } if P::MIDI_INPUT >= MidiConfig::Basic => out.try_push(NoteEndEvent(
                    clack_plugin::events::event_types::NoteEvent::new(
                        EventHeader::new_core(time, EventFlags::empty()),
                        voice_id.unwrap_or(-1),
                        0,
                        note as i16,
                        channel as i16,
                        0.0,
                    ),
                )),
                NoteEvent::PolyPressure {
                    timing: _,
                    voice_id,
                    channel,
                    note,
                    pressure,
                } if P::MIDI_OUTPUT >= MidiConfig::Basic => out.try_push(NoteExpressionEvent::new(
                    EventHeader::new_core(time, EventFlags::empty()),
                    voice_id.unwrap_or(-1),
                    0,
                    note as i16,
                    channel as i16,
                    pressure as f64,
                    NoteExpressionType::Pressure,
                )),
                NoteEvent::PolyVolume {
                    timing: _,
                    voice_id,
                    channel,
                    note,
                    gain,
                } if P::MIDI_OUTPUT >= MidiConfig::Basic => out.try_push(NoteExpressionEvent::new(
                    EventHeader::new_core(time, EventFlags::empty()),
                    voice_id.unwrap_or(-1),
                    0,
                    note as i16,
                    channel as i16,
                    gain as f64,
                    NoteExpressionType::Volume,
                )),
                NoteEvent::PolyPan {
                    timing: _,
                    voice_id,
                    channel,
                    note,
                    pan,
                } if P::MIDI_OUTPUT >= MidiConfig::Basic => out.try_push(NoteExpressionEvent::new(
                    EventHeader::new_core(time, EventFlags::empty()),
                    voice_id.unwrap_or(-1),
                    0,
                    note as i16,
                    channel as i16,
                    pan as f64,
                    NoteExpressionType::Pan,
                )),
                NoteEvent::PolyTuning {
                    timing: _,
                    voice_id,
                    channel,
                    note,
                    tuning,
                } if P::MIDI_OUTPUT >= MidiConfig::Basic => out.try_push(NoteExpressionEvent::new(
                    EventHeader::new_core(time, EventFlags::empty()),
                    voice_id.unwrap_or(-1),
                    0,
                    note as i16,
                    channel as i16,
                    tuning as f64,
                    NoteExpressionType::Tuning,
                )),
                NoteEvent::PolyVibrato {
                    timing: _,
                    voice_id,
                    channel,
                    note,
                    vibrato,
                } if P::MIDI_OUTPUT >= MidiConfig::Basic => out.try_push(NoteExpressionEvent::new(
                    EventHeader::new_core(time, EventFlags::empty()),
                    voice_id.unwrap_or(-1),
                    0,
                    note as i16,
                    channel as i16,
                    vibrato as f64,
                    NoteExpressionType::Vibrato,
                )),
                NoteEvent::PolyExpression {
                    timing: _,
                    voice_id,
                    channel,
                    note,
                    expression,
                } if P::MIDI_OUTPUT >= MidiConfig::Basic => out.try_push(NoteExpressionEvent::new(
                    EventHeader::new_core(time, EventFlags::empty()),
                    voice_id.unwrap_or(-1),
                    0,
                    note as i16,
                    channel as i16,
                    expression as f64,
                    NoteExpressionType::Expression,
                )),
                NoteEvent::PolyBrightness {
                    timing: _,
                    voice_id,
                    channel,
                    note,
                    brightness,
                } if P::MIDI_OUTPUT >= MidiConfig::Basic => out.try_push(NoteExpressionEvent::new(
                    EventHeader::new_core(time, EventFlags::empty()),
                    voice_id.unwrap_or(-1),
                    0,
                    note as i16,
                    channel as i16,
                    brightness as f64,
                    NoteExpressionType::Brightness,
                )),
                midi_event @ (NoteEvent::MidiChannelPressure { .. }
                | NoteEvent::MidiPitchBend { .. }
                | NoteEvent::MidiCC { .. }
                | NoteEvent::MidiProgramChange { .. })
                    if P::MIDI_OUTPUT >= MidiConfig::MidiCCs =>
                {
                    // NIH-plug already includes MIDI conversion functions, so we'll reuse those for
                    // the MIDI events
                    let midi_data = match midi_event.as_midi() {
                        Some(MidiResult::Basic(midi_data)) => midi_data,
                        Some(MidiResult::SysEx(_, _)) => unreachable!(
                            "Basic MIDI event read as SysEx, something's gone horribly wrong"
                        ),
                        None => unreachable!("Missing MIDI conversion for MIDI event"),
                    };

                    out.try_push(MidiEvent::new(
                        EventHeader::new_core(time, EventFlags::empty()),
                        0,
                        midi_data,
                    ))
                }
                NoteEvent::MidiSysEx { timing: _, message }
                    if P::MIDI_OUTPUT >= MidiConfig::Basic =>
                {
                    // SysEx is supported on the basic MIDI config so this is separate
                    let (padded_sysex_buffer, length) = message.to_buffer();
                    let padded_sysex_buffer = padded_sysex_buffer.borrow();
                    nih_debug_assert!(padded_sysex_buffer.len() >= length);
                    let sysex_buffer = &padded_sysex_buffer[..length];

                    out.try_push(MidiSysExEvent::new(
                        EventHeader::new_core(time, EventFlags::empty()),
                        0,
                        sysex_buffer,
                    ))
                }
                _ => {
                    nih_debug_assert_failure!(
                        "Invalid output event for the current MIDI_OUTPUT setting"
                    );
                    continue;
                }
            };

            nih_debug_assert!(push_successful.is_ok(), "Could not send note event");
        }
    }

    /// Handle an incoming CLAP event. The sample index is provided to support block splitting for
    /// sample accurate automation. [`input_events`][Self::input_events] must be cleared at the
    /// start of each process block.
    ///
    /// To save on mutex operations when handing MIDI events, the lock guard for the input events
    /// need to be passed into this function.
    ///
    /// If the event was a transport event and the `transport_info` argument is not `None`, then the
    /// pointer will be changed to point to the transport information from this event.
    pub fn handle_in_event(
        &self,
        event: &UnknownEvent,
        input_events: &mut AtomicRefMut<VecDeque<PluginNoteEvent<P>>>,
        transport_info: Option<&mut &TransportEvent>,
        current_sample_idx: usize,
        total_buffer_len: usize,
    ) {
        // Out of bounds events are clamped to the buffer's size
        let timing = clamp_input_event_timing(
            event.header().time() - current_sample_idx as u32,
            total_buffer_len as u32,
        );

        // We only support the Core event space for now
        let Some(event) = event.as_core_event() else { return };

        match event {
            CoreEventSpace::ParamValue(event) => {
                self.update_plain_value_by_hash(
                    event.param_id(),
                    ClapParamUpdate::PlainValueSet(event.value()),
                    self.current_buffer_config.load().map(|c| c.sample_rate),
                );

                // If the parameter supports polyphonic modulation, then the plugin needs to be
                // informed that the parameter has been monophonically automated. This allows the
                // plugin to update all of its polyphonic modulation values, since polyphonic
                // modulation acts as an offset to the monophonic value.
                if let Some(poly_modulation_id) = self.poly_mod_ids_by_hash.get(&event.param_id()) {
                    // The modulation offset needs to be normalized to account for modulated
                    // integer or enum parameters
                    let param_ptr = self.param_by_hash[&event.param_id()];
                    let normalized_value = event.value() as f32
                        / unsafe { param_ptr.step_count() }.unwrap_or(1) as f32;

                    input_events.push_back(NoteEvent::MonoAutomation {
                        timing,
                        poly_modulation_id: *poly_modulation_id,
                        normalized_value,
                    });
                }
            }
            CoreEventSpace::ParamMod(event) => {
                if event.note_id() != -1 && P::MIDI_INPUT >= MidiConfig::Basic {
                    match self.poly_mod_ids_by_hash.get(&event.param_id()) {
                        Some(poly_modulation_id) => {
                            // The modulation offset needs to be normalized to account for modulated
                            // integer or enum parameters
                            let param_ptr = self.param_by_hash[&event.param_id()];
                            let normalized_offset = event.amount() as f32
                                / unsafe { param_ptr.step_count() }.unwrap_or(1) as f32;

                            // The host may also add key and channel information here, but it may
                            // also pass -1. So not having that information here at all seems like
                            // the safest choice.
                            input_events.push_back(NoteEvent::PolyModulation {
                                timing,
                                voice_id: event.note_id(),
                                poly_modulation_id: *poly_modulation_id,
                                normalized_offset,
                            });

                            return;
                        }
                        None => nih_debug_assert_failure!(
                            "Polyphonic modulation sent for a parameter without a poly modulation \
                             ID"
                        ),
                    }
                }

                self.update_plain_value_by_hash(
                    event.param_id(),
                    ClapParamUpdate::PlainValueMod(event.amount()),
                    self.current_buffer_config.load().map(|c| c.sample_rate),
                );
            }
            CoreEventSpace::Transport(event) => {
                if let Some(transport_info) = transport_info {
                    *transport_info = event;
                }
            }
            CoreEventSpace::NoteOn(NoteOnEvent(event)) => {
                if P::MIDI_INPUT >= MidiConfig::Basic {
                    input_events.push_back(NoteEvent::NoteOn {
                        // When splitting up the buffer for sample accurate automation all events
                        // should be relative to the block
                        timing,
                        voice_id: if event.note_id() != -1 {
                            Some(event.note_id())
                        } else {
                            None
                        },
                        channel: event.channel() as u8,
                        note: event.key() as u8,
                        velocity: event.velocity() as f32,
                    });
                }
            }
            CoreEventSpace::NoteOff(NoteOffEvent(event)) => {
                if P::MIDI_INPUT >= MidiConfig::Basic {
                    input_events.push_back(NoteEvent::NoteOff {
                        timing,
                        voice_id: if event.note_id() != -1 {
                            Some(event.note_id())
                        } else {
                            None
                        },
                        channel: event.channel() as u8,
                        note: event.key() as u8,
                        velocity: event.velocity() as f32,
                    });
                }
            }
            CoreEventSpace::NoteChoke(NoteChokeEvent(event)) => {
                if P::MIDI_INPUT >= MidiConfig::Basic {
                    input_events.push_back(NoteEvent::Choke {
                        timing,
                        voice_id: if event.note_id() != -1 {
                            Some(event.note_id())
                        } else {
                            None
                        },
                        // FIXME: These values are also allowed to be -1, we need to support that
                        channel: event.channel() as u8,
                        note: event.key() as u8,
                    });
                }
            }
            CoreEventSpace::NoteExpression(event) => {
                if P::MIDI_INPUT >= MidiConfig::Basic {
                    // TODO: Add support for the other expression types
                    match event.expression_type() {
                        Some(NoteExpressionType::Pressure) => {
                            input_events.push_back(NoteEvent::PolyPressure {
                                timing,
                                voice_id: if event.note_id() != -1 {
                                    Some(event.note_id())
                                } else {
                                    None
                                },
                                channel: event.channel() as u8,
                                note: event.key() as u8,
                                pressure: event.value() as f32,
                            });
                        }
                        Some(NoteExpressionType::Volume) => {
                            input_events.push_back(NoteEvent::PolyVolume {
                                timing,
                                voice_id: if event.note_id() != -1 {
                                    Some(event.note_id())
                                } else {
                                    None
                                },
                                channel: event.channel() as u8,
                                note: event.key() as u8,
                                gain: event.value() as f32,
                            });
                        }
                        Some(NoteExpressionType::Pan) => {
                            input_events.push_back(NoteEvent::PolyPan {
                                timing,
                                voice_id: if event.note_id() != -1 {
                                    Some(event.note_id())
                                } else {
                                    None
                                },
                                channel: event.channel() as u8,
                                note: event.key() as u8,
                                // In CLAP this value goes from [0, 1] instead of [-1, 1]
                                pan: (event.value() as f32 * 2.0) - 1.0,
                            });
                        }
                        Some(NoteExpressionType::Tuning) => {
                            input_events.push_back(NoteEvent::PolyTuning {
                                timing,
                                voice_id: if event.note_id() != -1 {
                                    Some(event.note_id())
                                } else {
                                    None
                                },
                                channel: event.channel() as u8,
                                note: event.key() as u8,
                                tuning: event.value() as f32,
                            });
                        }
                        Some(NoteExpressionType::Vibrato) => {
                            input_events.push_back(NoteEvent::PolyVibrato {
                                timing,
                                voice_id: if event.note_id() != -1 {
                                    Some(event.note_id())
                                } else {
                                    None
                                },
                                channel: event.channel() as u8,
                                note: event.key() as u8,
                                vibrato: event.value() as f32,
                            });
                        }
                        Some(NoteExpressionType::Expression) => {
                            input_events.push_back(NoteEvent::PolyExpression {
                                timing,
                                voice_id: if event.note_id() != -1 {
                                    Some(event.note_id())
                                } else {
                                    None
                                },
                                channel: event.channel() as u8,
                                note: event.key() as u8,
                                expression: event.value() as f32,
                            });
                        }
                        Some(NoteExpressionType::Brightness) => {
                            input_events.push_back(NoteEvent::PolyBrightness {
                                timing,
                                voice_id: if event.note_id() != -1 {
                                    Some(event.note_id())
                                } else {
                                    None
                                },
                                channel: event.channel() as u8,
                                note: event.key() as u8,
                                brightness: event.value() as f32,
                            });
                        }
                        n => nih_debug_assert_failure!("Unhandled note expression ID {:?}", n),
                    }
                }
            }
            CoreEventSpace::Midi(event) => {
                // In the Basic note port type, we'll still handle note on, note off, and polyphonic
                // pressure events if the host sents us those. But we'll throw away any other MIDI
                // messages to stay consistent with the VST3 wrapper.

                match NoteEvent::from_midi(timing, &event.data()) {
                    Ok(
                        note_event @ (NoteEvent::NoteOn { .. }
                        | NoteEvent::NoteOff { .. }
                        | NoteEvent::PolyPressure { .. }),
                    ) if P::MIDI_INPUT >= MidiConfig::Basic => {
                        input_events.push_back(note_event);
                    }
                    Ok(note_event) if P::MIDI_INPUT >= MidiConfig::MidiCCs => {
                        input_events.push_back(note_event);
                    }
                    Ok(_) => (),
                    Err(n) => nih_debug_assert_failure!("Unhandled MIDI message type {}", n),
                };
            }
            CoreEventSpace::MidiSysEx(event) if P::MIDI_INPUT >= MidiConfig::Basic => {
                // `NoteEvent::from_midi` prints some tracing if parsing fails, which is not
                // necessarily an error
                if let Ok(note_event) = NoteEvent::from_midi(timing, event.data()) {
                    input_events.push_back(note_event);
                };
            }
            event => {
                nih_trace!("Unhandled CLAP event {:?}", event);
            }
        }
    }

    /// Get the plugin's state object, may be called by the plugin's GUI as part of its own preset
    /// management. The wrapper doesn't use these functions and serializes and deserializes directly
    /// the JSON in the relevant plugin API methods instead.
    pub fn get_state_object(&self) -> PluginState {
        unsafe {
            state::serialize_object::<P>(
                self.params.clone(),
                state::make_params_iter(&self.param_by_hash, &self.param_id_to_hash),
            )
        }
    }

    /// Update the plugin's internal state, called by the plugin itself from the GUI thread. To
    /// prevent corrupting data and changing parameters during processing the actual state is only
    /// updated at the end of the audio processing cycle.
    pub fn set_state_object_from_gui(&self, mut state: PluginState) {
        // Use a loop and timeouts to handle the super rare edge case when this function gets called
        // between a process call and the host disabling the plugin
        loop {
            if self.is_processing.load(Ordering::SeqCst) {
                // If the plugin is currently processing audio, then we'll perform the restore
                // operation at the end of the audio call. This involves sending the state to the
                // audio thread, having the audio thread handle the state restore at the very end of
                // the process function, and then sending the state back to this thread so it can be
                // deallocated without blocking the audio thread.
                match self
                    .updated_state_sender
                    .send_timeout(state, Duration::from_secs(1))
                {
                    Ok(_) => {
                        // As mentioned above, the state object will be passed back to this thread
                        // so we can deallocate it without blocking.
                        let state = self.updated_state_receiver.recv();
                        drop(state);
                        break;
                    }
                    Err(SendTimeoutError::Timeout(value)) => {
                        state = value;
                        continue;
                    }
                    Err(SendTimeoutError::Disconnected(_)) => {
                        nih_debug_assert_failure!("State update channel got disconnected");
                        return;
                    }
                }
            } else {
                // Otherwise we'll set the state right here and now, since this function should be
                // called from a GUI thread
                self.set_state_inner(&mut state);
                break;
            }
        }

        // After the state has been updated, notify the host about the new parameter values
        let task_posted = self.schedule_gui(Task::RescanParamValues);
        nih_debug_assert!(task_posted, "The task queue is full, dropping task...");
    }

    pub fn set_latency_samples(&self, samples: u32) {
        // Only make a callback if it's actually needed
        // XXX: For CLAP we could move this handling to the Plugin struct, but it may be worthwhile
        //      to keep doing it this way to stay consistent with VST3.
        let old_latency = self.current_latency.swap(samples, Ordering::SeqCst);
        if old_latency != samples {
            let task_posted = self.schedule_gui(Task::LatencyChanged);
            nih_debug_assert!(task_posted, "The task queue is full, dropping task...");
        }
    }

    pub fn set_current_voice_capacity(&self, capacity: u32) {
        match P::CLAP_POLY_MODULATION_CONFIG {
            Some(config) => {
                let clamped_capacity = capacity.clamp(1, config.max_voice_capacity);
                nih_debug_assert_eq!(
                    capacity,
                    clamped_capacity,
                    "The current voice capacity must be between 1 and the maximum capacity"
                );

                if clamped_capacity != self.current_voice_capacity.load(Ordering::Relaxed) {
                    self.current_voice_capacity
                        .store(clamped_capacity, Ordering::Relaxed);
                    let task_posted = self.schedule_gui(Task::VoiceInfoChanged);
                    nih_debug_assert!(task_posted, "The task queue is full, dropping task...");
                }
            }
            None => nih_debug_assert_failure!(
                "Configuring the current voice capacity is only possible when \
                 'ClapPlugin::CLAP_POLY_MODULATION_CONFIG' is set"
            ),
        }
    }
}

    /// Immediately set the plugin state. Returns `false` if the deserialization failed. The plugin
    /// state is set from a couple places, so this function aims to deduplicate that. Includes
    /// `permit_alloc()`s around the deserialization and initialization for the use case where
    /// `set_state_object_from_gui()` was called while the plugin is process audio.
    ///
    /// Implicitly emits `Task::ParameterValuesChanged`.
    ///
    /// # Notes
    ///
    /// `self.plugin` must _not_ be locked while calling this function or it will deadlock.
    pub fn set_state_inner(&self, state: &mut PluginState) -> bool {
        let audio_io_layout = self.current_audio_io_layout.load();
        let buffer_config = self.current_buffer_config.load();

        // FIXME: This is obviously not realtime-safe, but loading presets without doing this could
        //        lead to inconsistencies. It's the plugin's responsibility to not perform any
        //        realtime-unsafe work when the initialize function is called a second time if it
        //        supports runtime preset loading.  `state::deserialize_object()` normally never
        //        allocates, but if the plugin has persistent non-parameter data then its
        //        `deserialize_fields()` implementation may still allocate.
        let mut success = permit_alloc(|| unsafe {
            state::deserialize_object::<P>(
                state,
                self.params.clone(),
                state::make_params_getter(&self.param_by_hash, &self.param_id_to_hash),
                self.current_buffer_config.load().as_ref(),
            )
        });
        if !success {
            nih_debug_assert_failure!("Deserializing plugin state from a state object failed");
            return false;
        }

        // If the plugin was already initialized then it needs to be reinitialized
        if let Some(buffer_config) = buffer_config {
            // NOTE: This needs to be dropped after the `plugin` lock to avoid deadlocks
            let mut init_context = self.make_init_context();
            let mut plugin = self.plugin.lock();

            // See above
            success = permit_alloc(|| {
                plugin.initialize(&audio_io_layout, &buffer_config, &mut init_context)
            });
            if success {
                process_wrapper(|| plugin.reset());
            }
        }

        nih_debug_assert!(
            success,
            "Plugin returned false when reinitializing after loading state"
        );

        // Reinitialize the plugin after loading state so it can respond to the new parameter values
        let task_posted = self.schedule_gui(Task::ParameterValuesChanged);
        nih_debug_assert!(task_posted, "The task queue is full, dropping task...");

        // TODO: Right now there's no way to know if loading the state changed the GUI's size. We
        //       could keep track of the last known size and compare the GUI's current size against
        //       that but that also seems brittle.
        if self.editor_handle.lock().is_some() {
            self.request_resize();
        }

        success
    }

impl<'a, P: ClapPlugin> clack_plugin::plugin::Plugin<'a> for ClapWrapperAudioProcessor<'a, P> {
    type Shared = ClapWrapperShared<'a, P>;
    type MainThread = ClapWrapperMainThread<'a, P>;

    fn activate(
        host: HostAudioThreadHandle<'a>,
        _main_thread: &mut Self::MainThread,
        shared: &'a Self::Shared,
        audio_config: AudioConfiguration,
    ) -> Result<Self, PluginError> {
        let shared = shared.wrapper.clone();
        let audio_io_layout = shared.current_audio_io_layout.load();
        let buffer_config = BufferConfig {
            sample_rate: audio_config.sample_rate as f32,
            min_buffer_size: Some(audio_config.min_sample_count),
            max_buffer_size: audio_config.max_sample_count,
            process_mode: shared.current_process_mode.load(),
        };

        // Before initializing the plugin, make sure all smoothers are set the the default values
        for param in shared.param_by_hash.values() {
            unsafe {
                param.update_smoother(buffer_config.sample_rate, true);
            }
        }

        // NOTE: This needs to be dropped after the `plugin` lock to avoid deadlocks
        let mut init_context = shared.make_init_context();
        let mut plugin = shared.plugin.lock();

        if !plugin.initialize(&audio_io_layout, &buffer_config, &mut init_context) {
            return Err(PluginError::OperationFailed);
        }

        // NOTE: `Plugin::reset()` is called in `clap_plugin::start_processing()` instead of in
        //       this function

            // This preallocates enough space so we can transform all of the host's raw channel
            // pointers into a set of `Buffer` objects for the plugin's main and auxiliary IO
            *wrapper.buffer_manager.borrow_mut() =
                BufferManager::for_audio_io_layout(max_frames_count as usize, audio_io_layout);

        // Also store this for later, so we can reinitialize the plugin after restoring state
        shared.current_buffer_config.store(Some(buffer_config));

        Ok(Self { host, shared })
    }

    fn get_descriptor() -> Box<dyn clack_plugin::plugin::descriptor::PluginDescriptor> {
        todo!()
    }

    fn deactivate(self, _main_thread: &mut Self::MainThread) {
        self.shared.plugin.lock().deactivate();
    }

    fn start_processing(&mut self) -> Result<(), PluginError> {
        // We just need to keep track of our processing state so we can request a flush when
        // updating parameters from the GUI while the processing loop isn't running

        // Always reset the processing status when the plugin gets activated or deactivated
        self.shared.last_process_status.store(ProcessStatus::Normal);
        self.shared.is_processing.store(true, Ordering::SeqCst);

        // To be consistent with the VST3 wrapper, we'll also reset the buffers here in addition to
        // the dedicated `reset()` function.
        process_wrapper(|| self.shared.plugin.lock().reset());

        Ok(())
    }

    fn stop_processing(&mut self) {
        self.shared.is_processing.store(false, Ordering::SeqCst);
    }

    fn reset(&mut self, _main_thread: &mut Self::MainThread) {
        process_wrapper(|| self.shared.plugin.lock().reset());
    }

    fn process(
        &mut self,
        process: &Process,
        mut audio: Audio,
        events: Events,
    ) -> Result<clack_plugin::prelude::ProcessStatus, PluginError> {
        // Panic on allocations if the `assert_process_allocs` feature has been enabled, and make
        // sure that FTZ is set up correctly
        process_wrapper(|| {
            // We need to handle incoming automation and MIDI events. Since we don't support sample
            // accuration automation yet and there's no way to get the last event for a parameter,
            // we'll process every incoming event.
            let total_buffer_len = process.frames_count() as usize;

            let current_audio_io_layout = wrapper.current_audio_io_layout.load();
            let has_main_input = current_audio_io_layout.main_input_channels.is_some();
            let has_main_output = current_audio_io_layout.main_output_channels.is_some();
            let aux_input_start_idx = if has_main_input { 1 } else { 0 };
            let aux_output_start_idx = if has_main_output { 1 } else { 0 };

            // If `P::SAMPLE_ACCURATE_AUTOMATION` is set, then we'll split up the audio buffer into
            // chunks whenever a parameter change occurs
            let mut block_start = 0;
            let mut block_end = total_buffer_len;
            let mut event_start_idx = 0;

            // The host may send new transport information as an event. In that case we'll also
            // split the buffer.
            let mut transport_info = process.transport();

            let result = loop {
                let split_result = self.shared.handle_in_events_until(
                    events.input,
                    &mut transport_info,
                    block_start,
                    total_buffer_len,
                    event_start_idx,
                    |next_event| {
                        // Always split the buffer on transport information changes (tempo, time
                        // signature, or position changes), and also split on parameter value
                        // changes after the current sample if sample accurate automation is
                        // enabled
                        if P::SAMPLE_ACCURATE_AUTOMATION {
                            match next_event.as_core_event() {
                                Some(
                                    CoreEventSpace::Transport(_) | CoreEventSpace::ParamValue(_),
                                ) => true,
                                Some(CoreEventSpace::ParamMod(next_event)) => {
                                    // The buffer should not be split on polyphonic modulation
                                    // as those events will be converted to note events
                                    !(next_event.note_id() != -1
                                        && self
                                            .shared
                                            .poly_mod_ids_by_hash
                                            .contains_key(&next_event.param_id()))
                                }
                                _ => false,
                            }
                        } else {
                            matches!(
                                next_event.as_core_event(),
                                Some(CoreEventSpace::Transport(_))
                            )
                        }
                    },
                );

                // If there are any parameter changes after `block_start` and sample
                // accurate automation is enabled or the host sends new transport
                // information, then we'll process a new block just after that. Otherwise we can
                // process all audio until the end of the buffer.
                match split_result {
                    Some((next_param_change_sample_idx, next_param_change_event_idx)) => {
                        block_end = next_param_change_sample_idx;
                        event_start_idx = next_param_change_event_idx;
                    }
                    None => block_end = total_buffer_len,
                }

                // After processing the events we now know where/if the block should be split, and
                // we can start preparing audio processing
                let block_len = block_end - block_start;

                // The buffer manager preallocated buffer slices for all the IO and storage for any
                // axuiliary inputs.
                // TODO: The audio buffers have a latency field, should we use those?
                // TODO: Like with VST3, should we expose some way to access or set the silence/constant
                //       flags?
                let mut buffer_manager = wrapper.buffer_manager.borrow_mut();
                let buffers =
                    buffer_manager.create_buffers(block_start, block_len, |buffer_source| {
                        // Explicitly take plugins with no main output that does have auxiliary
                        // outputs into account. Shouldn't happen, but if we just start copying
                        // audio here then that would result in unsoundness.
                        if process.audio_outputs_count > 0
                            && !process.audio_outputs.is_null()
                            && !(*process.audio_outputs).data32.is_null()
                            && has_main_output
                        {
                            let audio_output = &*process.audio_outputs;
                            let ptrs = NonNull::new(audio_output.data32 as *mut *mut f32).unwrap();
                            let num_channels = audio_output.channel_count as usize;

                            *buffer_source.main_output_channel_pointers =
                                Some(ChannelPointers { ptrs, num_channels });
                        }

                        if process.audio_inputs_count > 0
                            && !process.audio_inputs.is_null()
                            && !(*process.audio_inputs).data32.is_null()
                            && has_main_input
                        {
                            let audio_input = &*process.audio_inputs;
                            let ptrs = NonNull::new(audio_input.data32 as *mut *mut f32).unwrap();
                            let num_channels = audio_input.channel_count as usize;

                            *buffer_source.main_input_channel_pointers =
                                Some(ChannelPointers { ptrs, num_channels });
                        }

                        if !process.audio_inputs.is_null() {
                            for (aux_input_no, aux_input_channel_pointers) in buffer_source
                                .aux_input_channel_pointers
                                .iter_mut()
                                .enumerate()
                            {
                                let aux_input_idx = aux_input_no + aux_input_start_idx;
                                if aux_input_idx > process.audio_inputs_count as usize {
                                    break;
                                }

                                let audio_input = &*process.audio_inputs.add(aux_input_idx);
                                match NonNull::new(audio_input.data32 as *mut *mut f32) {
                                    Some(ptrs) => {
                                        let num_channels = audio_input.channel_count as usize;

                                        *aux_input_channel_pointers =
                                            Some(ChannelPointers { ptrs, num_channels });
                                    }
                                    None => continue,
                                }
                            }
                        }

                        if !process.audio_outputs.is_null() {
                            for (aux_output_no, aux_output_channel_pointers) in buffer_source
                                .aux_output_channel_pointers
                                .iter_mut()
                                .enumerate()
                            {
                                let aux_output_idx = aux_output_no + aux_output_start_idx;
                                if aux_output_idx > process.audio_outputs_count as usize {
                                    break;
                                }

                                let audio_output = &*process.audio_outputs.add(aux_output_idx);
                                match NonNull::new(audio_output.data32 as *mut *mut f32) {
                                    Some(ptrs) => {
                                        let num_channels = audio_output.channel_count as usize;

                                        *aux_output_channel_pointers =
                                            Some(ChannelPointers { ptrs, num_channels });
                                    }
                                    None => continue,
                                }
                            }
                        }
                    });

                // If the host does not provide outputs or if it does not provide the required
                // number of channels (should not happen, but Ableton Live does this for bypassed
                // VST3 plugins) then we'll skip audio processing. In that case
                // `buffer_manager.create_buffers` will have set one or more of the output buffers
                // to empty slices since there is no storage to point them to. The auxiliary input
                // buffers always point to valid storage.
                let mut buffer_is_valid = true;
                for output_buffer_slice in buffers.main_buffer.as_slice_immutable().iter().chain(
                    buffers
                        .aux_outputs
                        .iter()
                        .flat_map(|buffer| buffer.as_slice_immutable().iter()),
                ) {
                    if output_buffer_slice.is_empty() {
                        buffer_is_valid = false;
                        break;
                    }
                }

                nih_debug_assert!(buffer_is_valid);

                // Some of the fields are left empty because CLAP does not provide this information,
                // but the methods on [`Transport`] can reconstruct these values from the other
                // fields
                let sample_rate = self
                    .shared
                    .current_buffer_config
                    .load()
                    .expect("Process call without prior initialization call")
                    .sample_rate;
                let mut transport = Transport::new(sample_rate);
                let context = &*transport_info;

                transport.playing = context.flags.contains(TransportFlags::IS_PLAYING);
                transport.recording = context.flags.contains(TransportFlags::IS_PLAYING);
                transport.preroll_active =
                    Some(context.flags.contains(TransportFlags::IS_WITHIN_PRE_ROLL));
                if context.flags.contains(TransportFlags::HAS_TEMPO) {
                    transport.tempo = Some(context.tempo);
                }
                if context.flags.contains(TransportFlags::HAS_TIME_SIGNATURE) {
                    transport.time_sig_numerator = Some(context.time_signature_numerator as i32);
                    transport.time_sig_denominator =
                        Some(context.time_signature_denominator as i32);
                }
                if context.flags.contains(TransportFlags::HAS_BEATS_TIMELINE) {
                    let beats = context.song_pos_beats.to_float();

                    // This is a bit messy, but we'll try to compensate for the block splitting.
                    // We can't use the functions on the transport information object for this
                    // because we don't have any sample information.
                    if P::SAMPLE_ACCURATE_AUTOMATION
                        && block_start > 0
                        && context.flags.contains(TransportFlags::HAS_TEMPO)
                    {
                        transport.pos_beats = Some(
                            beats
                                + (block_start as f64 / sample_rate as f64 / 60.0 * context.tempo),
                        );
                    } else {
                        transport.pos_beats = Some(beats);
                    }
                }
                if context.flags.contains(TransportFlags::HAS_SECONDS_TIMELINE) {
                    let seconds = context.song_pos_seconds.to_float();

                    // Same here
                    if P::SAMPLE_ACCURATE_AUTOMATION
                        && block_start > 0
                        && context.flags.contains(TransportFlags::HAS_TEMPO)
                    {
                        transport.pos_seconds =
                            Some(seconds + (block_start as f64 / sample_rate as f64));
                    } else {
                        transport.pos_seconds = Some(seconds);
                    }
                }
                // TODO: CLAP does not mention whether this is behind a flag or not
                if P::SAMPLE_ACCURATE_AUTOMATION && block_start > 0 {
                    transport.bar_start_pos_beats = match transport.bar_start_pos_beats() {
                        Some(updated) => Some(updated),
                        None => Some(context.bar_start.to_float()),
                    };
                    transport.bar_number = match transport.bar_number() {
                        Some(updated) => Some(updated),
                        None => Some(context.bar_number),
                    };
                } else {
                    transport.bar_start_pos_beats = Some(context.bar_start.to_float());
                    transport.bar_number = Some(context.bar_number);
                }
                // TODO: They also aren't very clear about this, but presumably if the loop is
                //       active and the corresponding song transport information is available then
                //       this is also available
                if context
                    .flags
                    .contains(TransportFlags::IS_LOOP_ACTIVE & TransportFlags::HAS_BEATS_TIMELINE)
                {
                    transport.loop_range_beats = Some((
                        context.loop_start_beats.to_float(),
                        context.loop_end_beats.to_float(),
                    ));
                }
                if context
                    .flags
                    .contains(TransportFlags::IS_LOOP_ACTIVE & TransportFlags::HAS_SECONDS_TIMELINE)
                {
                    transport.loop_range_seconds = Some((
                        context.loop_start_seconds.to_float(),
                        context.loop_end_seconds.to_float(),
                    ));
                }

                let result = if buffer_is_valid {
                    let mut plugin = self.shared.plugin.lock();
                    // SAFETY: Shortening these borrows is safe as even if the plugin overwrites the
                    //         slices (which it cannot do without using unsafe code), then they
                    //         would still be reset on the next iteration
                    let mut aux = unsafe {
                        AuxiliaryBuffers {
                            inputs: buffers.aux_inputs,
                            outputs: buffers.aux_outputs,
                        }
                    };
                    let mut context = self.shared.make_process_context(transport);
                    let result = plugin.process(buffers.main_buffer, &mut aux, &mut context);
                    self.shared.last_process_status.store(result);
                    result
                } else {
                    ProcessStatus::Normal
                };

                let clap_result = match result {
                    ProcessStatus::Error(err) => {
                        nih_debug_assert_failure!("Process error: {}", err);

                        return Err(PluginError::OperationFailed); // TODO: place the error here
                    }
                    ProcessStatus::Normal => {
                        clack_plugin::prelude::ProcessStatus::ContinueIfNotQuiet
                    }
                    ProcessStatus::Tail(_) => clack_plugin::prelude::ProcessStatus::Continue,
                    ProcessStatus::KeepAlive => clack_plugin::prelude::ProcessStatus::Continue,
                };

                // After processing audio, send all spooled events to the host. This include note
                // events.
                self.shared
                    .handle_out_events(events.output, block_start, total_buffer_len);

                // If our block ends at the end of the buffer then that means there are no more
                // unprocessed (parameter) events. If there are more events, we'll just keep going
                // through this process until we've processed the entire buffer.
                if block_end == total_buffer_len {
                    break clap_result;
                } else {
                    block_start = block_end;
                }
            };

            // After processing audio, we'll check if the editor has sent us updated plugin state.
            // We'll restore that here on the audio thread to prevent changing the values during the
            // process call and also to prevent inconsistent state when the host also wants to load
            // plugin state.
            // FIXME: Zero capacity channels allocate on receiving, find a better alternative that
            //        doesn't do that
            let updated_state = permit_alloc(|| self.shared.updated_state_receiver.try_recv());
            if let Ok(mut state) = updated_state {
                wrapper.set_state_inner(&mut state);

                // We'll pass the state object back to the GUI thread so deallocation can happen
                // there without potentially blocking the audio thread
                if let Err(err) = self.shared.updated_state_sender.send(state) {
                    nih_debug_assert_failure!(
                        "Failed to send state object back to GUI thread: {}",
                        err
                    );
                };
            }

            Ok(result)
        })
    }

    fn declare_extensions(builder: &mut PluginExtensions<Self>, shared: &Self::Shared) {
        builder
            .register::<PluginAudioPortsConfig>()
            .register::<PluginAudioPorts>()
            .register::<PluginLatency>()
            .register::<PluginParams>()
            .register::<PluginRender>()
            .register::<clack_extensions::state::PluginState>()
            .register::<PluginTail>();

        if shared.wrapper.editor.borrow().is_some() {
            builder.register::<PluginGui>();
        }

        if P::MIDI_INPUT >= MidiConfig::Basic || P::MIDI_OUTPUT >= MidiConfig::Basic {
            builder.register::<PluginNotePorts>();
        }

        if P::CLAP_POLY_MODULATION_CONFIG.is_some() {
            builder.register::<PluginVoiceInfo>();
        }
    }
}

impl<'a, P: ClapPlugin> PluginMainThread<'a, ClapWrapperShared<'a, P>>
    for ClapWrapperMainThread<'a, P>
{
    fn new(
        host: HostMainThreadHandle<'a>,
        shared: &'a ClapWrapperShared<'a, P>,
    ) -> Result<Self, PluginError> {
        Ok(Self {
            host,
            shared: shared.wrapper.clone(),
        })
    }

    fn on_main_thread(&mut self) {
        while let Some(task) = self.shared.tasks.pop() {
            self.shared.execute(task, true);
        }
    }
}
impl<'a, P: ClapPlugin> PluginAudioPortsConfigImpl for ClapWrapperMainThread<'a, P> {
    fn count(&self) -> u32 {
            P::AUDIO_IO_LAYOUTS.len() as u32
    }

    fn get(&self, index: u32, writer: &mut AudioPortConfigWriter) {

        // This function directly maps to `P::AUDIO_IO_LAYOUTS`, and we thus also don't need to
        // access the `wrapper` instance
        match P::AUDIO_IO_LAYOUTS.get(index as usize) {
            Some(audio_io_layout) => {
                let name = audio_io_layout.name();

                let main_input_channels = audio_io_layout.main_input_channels.map(NonZeroU32::get);
                let main_output_channels =
                    audio_io_layout.main_output_channels.map(NonZeroU32::get);
                let input_port_type = match main_input_channels {
                    Some(1) => CLAP_PORT_MONO.as_ptr(),
                    Some(2) => CLAP_PORT_STEREO.as_ptr(),
                    _ => std::ptr::null(),
                };
                let output_port_type = match main_output_channels {
                    Some(1) => CLAP_PORT_MONO.as_ptr(),
                    Some(2) => CLAP_PORT_STEREO.as_ptr(),
                    _ => std::ptr::null(),
                };

                *config = std::mem::zeroed();

                let config = &mut *config;
                config.id = index;
                strlcpy(&mut config.name, &name);
                config.input_port_count = (if main_input_channels.is_some() { 1 } else { 0 }
                    + audio_io_layout.aux_input_ports.len())
                    as u32;
                config.output_port_count = (if main_output_channels.is_some() { 1 } else { 0 }
                    + audio_io_layout.aux_output_ports.len())
                    as u32;
                config.has_main_input = main_input_channels.is_some();
                config.main_input_channel_count = main_input_channels.unwrap_or_default();
                config.main_input_port_type = input_port_type;
                config.has_main_output = main_output_channels.is_some();
                config.main_output_channel_count = main_output_channels.unwrap_or_default();
                config.main_output_port_type = output_port_type;

                true
            }
            None => {
                nih_debug_assert_failure!(
                    "Host tried to query out of bounds audio port config {}",
                    index
                );

                false
            }
        }
    }

    fn select(&mut self, config_id: u32) -> Result<(), AudioPortConfigSelectError> {
        // We use the vector indices for the config ID
        match P::AUDIO_IO_LAYOUTS.get(config_id as usize) {
            Some(audio_io_layout) => {
                wrapper.current_audio_io_layout.store(*audio_io_layout);

                Ok(())
            }
            None => {
                nih_debug_assert_failure!(
                    "Host tried to select out of bounds audio port config {}",
                    config_id
                );

                Err(AudioPortConfigSelectError)
            }
        }
    }
}

impl<'a, P: ClapPlugin> PluginAudioPortsImpl for ClapWrapperMainThread<'a, P> {
    fn count(&self, is_input: bool) -> u32 {
        let audio_io_layout = self.shared.current_audio_io_layout.load();
        if is_input {
            let main_ports = if audio_io_layout.main_input_channels.is_some() {
                1
            } else {
                0
            };
            let aux_ports = audio_io_layout.aux_input_ports.len();

            (main_ports + aux_ports) as u32
        } else {
            let main_ports = if audio_io_layout.main_output_channels.is_some() {
                1
            } else {
                0
            };
            let aux_ports = audio_io_layout.aux_output_ports.len();

            (main_ports + aux_ports) as u32
        }
    }

    fn get(&self, is_input: bool, index: u32, writer: &mut AudioPortInfoWriter) {
        let num_input_ports = <Self as PluginAudioPortsImpl>::count(self, true);
        let num_output_ports = <Self as PluginAudioPortsImpl>::count(self, false);
        if (is_input && index >= num_input_ports) || (!is_input && index >= num_output_ports) {
            nih_debug_assert_failure!(
                "Host tried to query information for out of bounds audio port {} (input: {})",
                index,
                is_input
            );

            return;
        }

        let current_audio_io_layout = self.shared.current_audio_io_layout.load();
        let has_main_input = current_audio_io_layout.main_input_channels.is_some();
        let has_main_output = current_audio_io_layout.main_output_channels.is_some();

        // Whether this port is a main port or an auxiliary (sidechain) port
        let is_main_port =
            index == 0 && ((is_input && has_main_input) || (!is_input && has_main_output));

        // We'll number the ports in a linear order from `0..num_input_ports` and
        // `num_input_ports..(num_input_ports + num_output_ports)`
        let stable_id = if is_input {
            index
        } else {
            index + num_input_ports
        };
        let pair_stable_id = match (is_input, is_main_port) {
            // Ports are named linearly with inputs coming before outputs, so this is the index of
            // the first output port
            (true, true) if has_main_output => Some(num_input_ports),
            (false, true) if has_main_input => Some(0),
            _ => None,
        };

        let channel_count = match (index, is_input) {
            (0, true) if has_main_input => {
                current_audio_io_layout.main_input_channels.unwrap().get()
            }
            (0, false) if has_main_output => {
                current_audio_io_layout.main_output_channels.unwrap().get()
            }
            // `index` is off by one for the auxiliary ports if the plugin has a main port
            (n, true) if has_main_input => {
                current_audio_io_layout.aux_input_ports[n as usize - 1].get()
            }
            (n, false) if has_main_output => {
                current_audio_io_layout.aux_output_ports[n as usize - 1].get()
            }
            (n, true) => current_audio_io_layout.aux_input_ports[n as usize].get(),
            (n, false) => current_audio_io_layout.aux_output_ports[n as usize].get(),
        };

        let port_type = match channel_count {
            1 => Some(AudioPortType::MONO),
            2 => Some(AudioPortType::STEREO),
            _ => None,
        };

        let name = match (is_input, is_main_port) {
            (true, true) =>  &current_audio_io_layout.main_input_name(),
            (false, true) => &current_audio_io_layout.main_output_name(),
            (true, false) => {
                let aux_input_idx = if has_main_input { index - 1 } else { index };
                &current_audio_io_layout
                .aux_input_name(aux_input_idx)
                .expect("Out of bounds auxiliary input port")
            }
            (false, false) => {
                let aux_output_idx = if has_main_output { index - 1 } else { index };
                &current_audio_io_layout
                .aux_output_name(aux_output_idx)
                .expect("Out of bounds auxiliary output port")
            }
        };

        writer.set(&AudioPortInfoData {
            name: name.as_bytes(),
            id: stable_id,
            flags: if is_main_port {
                AudioPortFlags::IS_MAIN
            } else {
                AudioPortFlags::empty()
            },
            channel_count,
            port_type,
            in_place_pair: pair_stable_id,
        });
    }
}

impl<'a, P: ClapPlugin> PluginGuiImpl for ClapWrapperMainThread<'a, P> {
    fn is_api_supported(&self, api: GuiApiType, is_floating: bool) -> bool {
        // We don't do standalone floating windows
        if is_floating {
            return false;
        }

        #[cfg(all(target_family = "unix", not(target_os = "macos")))]
        if api == GuiApiType::X11 {
            return true;
        }
        #[cfg(target_os = "macos")]
        if api == GuiApiType::COCOA {
            return true;
        }
        #[cfg(target_os = "windows")]
        if api == GuiApiType::WIN32 {
            return true;
        }

        false
    }

    fn get_preferred_api(&self) -> Option<(GuiApiType<'static>, bool)> {
        #[cfg(all(target_family = "unix", not(target_os = "macos")))]
        {
            Some((GuiApiType::X11, false))
        }
        #[cfg(target_os = "macos")]
        {
            Some((GuiApiType::COCOA, false))
        }
        #[cfg(target_os = "windows")]
        {
            Some((GuiApiType::WIN32, false))
        }
    }

    fn create(&mut self, api: GuiApiType, is_floating: bool) -> Result<(), GuiError> {
        // Double check this in case the host didn't
        if !self.is_api_supported(api, is_floating) {
            return Err(GuiError::CreateError);
        }

        // In CLAP creating the editor window and embedding it in another window are separate, and
        // those things are one and the same in our framework. So we'll just pretend we did
        // something here.

        let editor_handle = self.shared.editor_handle.lock();
        if editor_handle.is_none() {
            Ok(())
        } else {
            nih_debug_assert_failure!("Tried creating editor while the editor was already active");
            Err(GuiError::CreateError)
        }
    }

    fn destroy(&mut self) {
        let mut editor_handle = self.shared.editor_handle.lock();
        if editor_handle.is_some() {
            *editor_handle = None;
        } else {
            nih_debug_assert_failure!("Tried destroying editor while the editor was not active");
        }
    }

    fn set_scale(&mut self, scale: f64) -> Result<(), GuiError> {
        // On macOS scaling is done by the OS, and all window sizes are in logical pixels
        if cfg!(target_os = "macos") {
            nih_debug_assert_failure!("Ignoring host request to set explicit DPI scaling factor");
            return Err(GuiError::SetScaleError);
        }

        if self
            .shared
            .editor
            .borrow()
            .as_ref()
            .unwrap()
            .lock()
            .set_scale_factor(scale as f32)
        {
            self.shared
                .editor_scaling_factor
                .store(scale as f32, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        } else {
            Err(GuiError::SetScaleError)
        }
    }

    fn get_size(&mut self) -> Option<GuiSize> {
        // For macOS the scaling factor is always 1
        let (unscaled_width, unscaled_height) =
            self.shared.editor.borrow().as_ref().unwrap().lock().size();
        let scaling_factor = self.shared.editor_scaling_factor.load(Ordering::Relaxed);

        Some(GuiSize {
            width: (unscaled_width as f32 * scaling_factor).round() as u32,
            height: (unscaled_height as f32 * scaling_factor).round() as u32,
        })
    }

    fn can_resize(&self) -> bool {
        // TODO: Implement Host->Plugin GUI resizing
        false
    }

    fn get_resize_hints(&self) -> Option<GuiResizeHints> {
        // TODO: Implement Host->Plugin GUI resizing
        None
    }

    fn adjust_size(&mut self, _size: GuiSize) -> Option<GuiSize> {
        // TODO: Implement Host->Plugin GUI resizing
        None
    }

    fn set_size(&mut self, size: GuiSize) -> Result<(), GuiError> {
        // TODO: Implement Host->Plugin GUI resizing
        // TODO: The host will also call this if an asynchronous (on Linux) resize request fails
        let (unscaled_width, unscaled_height) =
            self.shared.editor.borrow().as_ref().unwrap().lock().size();
        let scaling_factor = self.shared.editor_scaling_factor.load(Ordering::Relaxed);
        let (editor_width, editor_height) = (
            (unscaled_width as f32 * scaling_factor).round() as u32,
            (unscaled_height as f32 * scaling_factor).round() as u32,
        );

        if size.width == editor_width && size.height == editor_height {
            Ok(())
        } else {
            Err(GuiError::SetScaleError)
        }
    }

    fn set_parent(&mut self, window: Window) -> Result<(), GuiError> {
        let mut editor_handle = self.shared.editor_handle.lock();
        if editor_handle.is_none() {
            let handle = window.raw_window_handle();

            // This extension is only exposed when we have an editor
            *editor_handle = Some(self.shared.editor.borrow().as_ref().unwrap().lock().spawn(
                ParentWindowHandle { handle },
                self.shared.clone().make_gui_context(),
            ));

            Ok(())
        } else {
            nih_debug_assert_failure!(
                "Host tried to attach editor while the editor is already attached"
            );

            Err(GuiError::SetParentError)
        }
    }

    fn set_transient(&mut self, _window: Window) -> Result<(), GuiError> {
        // This is only relevant for floating windows
        Err(GuiError::SetTransientError)
    }

    fn suggest_title(&mut self, _title: &str) {
        // This is only relevant for floating windows
    }

    fn show(&mut self) -> Result<(), GuiError> {
        // TODO: Does this get used? Is this only for the free-standing window extension? (which we
        //       don't implement) This wouldn't make any sense for embedded editors.
        Err(GuiError::ShowError)
    }

    fn hide(&mut self) -> Result<(), GuiError> {
        // TODO: Same as the above
        Err(GuiError::HideError)
    }
}

impl<'a, P: ClapPlugin> PluginLatencyImpl for ClapWrapperMainThread<'a, P> {
    fn get(&mut self) -> u32 {
        self.shared.current_latency.load(Ordering::SeqCst)
    }
}

impl<'a, P: ClapPlugin> PluginNotePortsImpl for ClapWrapperMainThread<'a, P> {
    fn count(&self, is_input: bool) -> u32 {
        match is_input {
            true if P::MIDI_INPUT >= MidiConfig::Basic => 1,
            false if P::MIDI_OUTPUT >= MidiConfig::Basic => 1,
            _ => 0,
        }
    }

    fn get(&self, is_input: bool, index: u32, writer: &mut NotePortInfoWriter) {
        match (index, is_input) {
            (0, true) if P::MIDI_INPUT >= MidiConfig::Basic => writer.set(&NotePortInfoData {
                id: 0,
                // NOTE: REAPER won't send us SysEx if we don't support the MIDI dialect
                // TODO: Implement MPE (would just be a toggle for the plugin to expose it) and MIDI2
                supported_dialects: NoteDialects::CLAP | NoteDialects::MIDI,
                preferred_dialect: Some(NoteDialect::Clap),
                name: b"Note Input",
            }),
            (0, false) if P::MIDI_OUTPUT >= MidiConfig::Basic => writer.set(&NotePortInfoData {
                id: 0,
                // If `P::MIDI_OUTPUT < MidiConfig::MidiCCs` we'll throw away MIDI CCs, pitch bend
                // messages, and other messages that are not basic note on, off and polyphonic
                // pressure messages. This way the behavior is the same as the VST3 wrapper.
                supported_dialects: NoteDialects::CLAP | NoteDialects::MIDI,
                preferred_dialect: Some(NoteDialect::Clap),
                name: b"Note Output",
            }),
            _ => {}
        }
    }
}

impl<'a, P: ClapPlugin> PluginMainThreadParams for ClapWrapperMainThread<'a, P> {
    fn count(&self) -> u32 {
        self.shared.param_hashes.len() as u32
    }

    fn get_info(&self, param_index: u32, info: &mut ParamInfoWriter) {
        if param_index > PluginMainThreadParams::count(self) {
            return;
        }

        let param_hash = self.shared.param_hashes[param_index as usize];
        let param_group = &self.shared.param_group_by_hash[&param_hash];
        let param_ptr = self.shared.param_by_hash[&param_hash];
        let default_value = unsafe { param_ptr.default_normalized_value() };
        let step_count = unsafe { param_ptr.step_count() };
        let flags = unsafe { param_ptr.flags() };
        let automatable = !flags.contains(ParamFlags::NON_AUTOMATABLE);
        let hidden = flags.contains(ParamFlags::HIDDEN);
        let is_bypass = flags.contains(ParamFlags::BYPASS);

        // TODO: We don't use the cookies at this point. In theory this would be faster than the ID
        //       hashmap lookup, but for now we'll stay consistent with the VST3 implementation.

        let mut param_info = ParamInfoData {
            id: param_hash,
            flags: ParamInfoFlags::empty(),
            cookie: Default::default(),
            name: unsafe { param_ptr.name() },
            module: &param_group,

            // We don't use the actual minimum and maximum values here because that would not scale
            // with skewed integer ranges. Instead, just treat all parameters as `[0, 1]` normalized
            // parameters multiplied by the step size.
            min_value: 0.0,

            // Stepped parameters are unnormalized float parameters since there's no separate step
            // range option
            // TODO: This should probably be encapsulated in some way so we don't forget about this in one place
            max_value: step_count.unwrap_or(1) as f64,
            default_value: default_value as f64 * step_count.unwrap_or(1) as f64,
        };

        // TODO: Somehow expose per note/channel/port modulation
        if automatable && !hidden {
            param_info.flags |= ParamInfoFlags::IS_AUTOMATABLE | ParamInfoFlags::IS_MODULATABLE;

            if self.shared.poly_mod_ids_by_hash.contains_key(&param_hash) {
                param_info.flags |= ParamInfoFlags::IS_MODULATABLE_PER_NOTE_ID;
            }
        }
        if hidden {
            param_info.flags |= ParamInfoFlags::IS_HIDDEN | ParamInfoFlags::IS_READONLY;
        }
        if is_bypass {
            param_info.flags |= ParamInfoFlags::IS_BYPASS
        }
        if step_count.is_some() {
            param_info.flags |= ParamInfoFlags::IS_STEPPED
        }

        info.set(&param_info);
    }

    fn get_value(&self, param_id: u32) -> Option<f64> {
        self.shared
            .param_by_hash
            .get(&param_id)
            .map(|param_ptr| unsafe {
                param_ptr.modulated_normalized_value() as f64
                    * param_ptr.step_count().unwrap_or(1) as f64
            })
    }

    fn value_to_text(
        &self,
        param_id: u32,
        value: f64,
        writer: &mut ParamDisplayWriter,
    ) -> std::fmt::Result {
        use core::fmt::Write;

        let Some(param_ptr) = self.shared.param_by_hash.get(&param_id) else { return Ok(()) };

        // TODO: we should pass the writer straight to normalized_value_to_string instead of allocating a String each time
        let str = unsafe {
            param_ptr.normalized_value_to_string(
                value as f32 / param_ptr.step_count().unwrap_or(1) as f32,
                true,
            )
        };

        writer.write_str(&str)
    }

    fn text_to_value(&self, param_id: u32, text: &str) -> Option<f64> {
        let param_ptr = self.shared.param_by_hash.get(&param_id)?;

        unsafe {
            let normalized_value = param_ptr.string_to_normalized_value(text)? as f64;
            Some(normalized_value * (param_ptr.step_count().unwrap_or(1)) as f64)
        }
    }

    fn flush(
        &mut self,
        input_parameter_changes: &InputEvents,
        output_parameter_changes: &mut OutputEvents,
    ) {
        self.shared.handle_in_events(input_parameter_changes, 0, 0);
        self.shared
            .handle_out_events(output_parameter_changes, 0, 0);
    }
}

impl<'a, P: ClapPlugin> PluginParamsImpl for ClapWrapperAudioProcessor<'a, P> {
    fn flush(
        &mut self,
        input_parameter_changes: &InputEvents,
        output_parameter_changes: &mut OutputEvents,
    ) {
        self.shared.handle_in_events(input_parameter_changes, 0, 0);
        self.shared
            .handle_out_events(output_parameter_changes, 0, 0);
    }
}

    unsafe extern "C" fn ext_remote_controls_count(plugin: *const clap_plugin) -> u32 {
        check_null_ptr!(0, plugin, (*plugin).plugin_data);
        let wrapper = &*((*plugin).plugin_data as *const Self);

        wrapper.remote_control_pages.len() as u32
    }

    unsafe extern "C" fn ext_remote_controls_get(
        plugin: *const clap_plugin,
        page_index: u32,
        page: *mut clap_remote_controls_page,
    ) -> bool {
        check_null_ptr!(false, plugin, (*plugin).plugin_data, page);
        let wrapper = &*((*plugin).plugin_data as *const Self);

        nih_debug_assert!(page_index as usize <= wrapper.remote_control_pages.len());
        match wrapper.remote_control_pages.get(page_index as usize) {
            Some(p) => {
                *page = *p;
                true
            }
            None => false,
        }
    }

impl<'a, P: ClapPlugin> PluginRenderImpl for ClapWrapperMainThread<'a, P> {
    fn has_hard_realtime_requirement(&self) -> bool {
        P::HARD_REALTIME_ONLY
    }

    fn set(&mut self, mode: RenderMode) -> Result<(), PluginRenderError> {
        self.shared.current_process_mode.store(match mode {
            RenderMode::Realtime => ProcessMode::Realtime,
            RenderMode::Offline => ProcessMode::Offline,
        });

        Ok(())
    }
}

impl<'a, P: ClapPlugin> PluginStateImpl for ClapWrapperMainThread<'a, P> {
    fn save(&mut self, output: &mut OutputStream) -> Result<(), PluginError> {
        use std::io::Write;

        let serialized = unsafe {
            state::serialize_json::<P>(
                self.shared.params.clone(),
                state::make_params_iter(&self.shared.param_by_hash, &self.shared.param_id_to_hash),
            )
            .map_err(|e| PluginError::Custom(e.into()))?
        };

        // CLAP does not provide a way to tell how much data there is left in a stream, so
        // we need to prepend it to our actual state data.
        let length_bytes = (serialized.len() as u64).to_le_bytes();
        output.write_all(&length_bytes)?;

        // TODO: pass writer directly to serialize_json instead of allocating an extra Vec
        output.write_all(&serialized)?;

        nih_trace!("Saved state ({} bytes)", serialized.len());

        Ok(())
    }

    fn load(&mut self, input: &mut InputStream) -> Result<(), PluginError> {
        // CLAP does not have a way to tell how much data there is left in a stream, so we've
        // prepended the size in front of our JSON state
        let mut length_bytes = [0u8; 8];
        input.read(&mut length_bytes)?;
        // TODO: this isn't useful anymore, but removing it is a breaking change
        let length = u64::from_le_bytes(length_bytes);

        let mut read_buffer: Vec<u8> = Vec::with_capacity(length as usize);
        input.read_to_end(&mut read_buffer)?;

        match state::deserialize_json(&read_buffer) {
            Some(mut state) => {
                let success = wrapper.set_state_inner(&mut state);
                if success {
                    nih_trace!("Loaded state ({} bytes)", read_buffer.len());
                }

                success
            }
            None => false,
        }
    }
}

impl<'a, P: ClapPlugin> PluginTailImpl for ClapWrapperAudioProcessor<'a, P> {
    fn get(&self) -> TailLength {
        // TODO: remove useless Atomic
        match self.shared.last_process_status.load() {
            ProcessStatus::Tail(samples) => TailLength::Finite(samples),
            ProcessStatus::KeepAlive => TailLength::Infinite,
            _ => TailLength::Finite(0),
        }
    }
}

impl<'a, P: ClapPlugin> PluginVoiceInfoImpl for ClapWrapperMainThread<'a, P> {
    fn get(&self) -> Option<VoiceInfo> {
        P::CLAP_POLY_MODULATION_CONFIG.map(|config| VoiceInfo {
            voice_count: self.shared.current_voice_capacity.load(Ordering::Relaxed),
            voice_capacity: config.max_voice_capacity,
            flags: if config.supports_overlapping_voices {
                VoiceInfoFlags::SUPPORTS_OVERLAPPING_NOTES
            } else {
                VoiceInfoFlags::empty()
            },
        })
    }
}
