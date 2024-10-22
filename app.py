# streamlit app for chatbot using langchain, openai, pinecone, and huggingface to generate PAL scripts
import os
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
import ast
import tiktoken
import dotenv


dotenv.load_dotenv()

pc_api_key = os.environ['PINECONE']

gpt_api_key = os.environ['OPENAI']

# Set your OpenAI API key here
client = OpenAI(
    # This is the default and can be omitted
    api_key=gpt_api_key,
)

# Pinecone setup
pinecone_api_key = pc_api_key
pinecone_index_name = 'Pal3-Scripting'.lower()
pinecone_environment = 'aws-starter'  # adjust based on your location
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)

index = pc.Index(pinecone_index_name)

TEMPERATURE=0.0


glossary = {
    "MoveToObject": "This activity moves the robot arm to the desired target position and detects the target using the touchdown mechanism. Automatically takes care of the approach path to the target (e.g. opening the agitator lid or pulling out stack drawers). Must always be used in pair with LeaveObject",
    "PenetrateObject": "The tool attached to the PALHead penetrates the target object at the given speed to the indicated depth. The depth is measured from the touchdown position. The needle guide is retracted by the Needle Guide Drive.",
    "PenetrateWithBottomSense": "The PALTool attached to the PALHead penetrates the target object at the given speed down to the bottom of the object. The bottom is detected with the needle tip.",
    "Depenetrate": "The PALTool attached to the PALHead depenetrates from any object and remains directly above it.",
    "LeaveObject": "The PALTool attached to the PALHead depenetrates from any object and performs the necessary steps to move away from the object (i.e. stripping a vial off the magnet ring, closing stack drawers)",
    "MoveToHome": "This activity moves the PALHead to the user definable home position.",
    "ChangeTool": "This activity parks the currently attached PALTool and mounts the selected PALTool to into the PALHead. Please note: The position where to park the currently attached PALTool within the Park Station is handled automatically. ChangeTool should always be executed at the beginning of a procedure to ensure the proper tool is used in a workflow. A procedure can contain multiple ChangeTool activities.",
    "TransportVial": "This activity transports 2mL, 10mL and 20mL magnetic cap vials by attaching them to the NeedleGuide and stripping them off at the destination object. 10mL & 20mL vials require that the large magnet ring is mounted on the Needle Guide. It is possible to modify the home position of the vial with each transportation step.",
    "TransportVialHome": "This activity transports a vial back to its home position. By default the home position is where the vial has been taken away at the beginning, but the home position can be modified during transport.",
    "VortexVial": "This activity transports a vial from source to the VortexMixer. The vial is then vortexed while the Needle Guide remains above the VortexMixer and holds the vial in place. Afterwards, the vial can be transported back using TransportVialHome() or transported elsewhere using TransportVial().",
    "ReadBarcode": "This activity reads the barcode of a vial. It transports it to the barcode reader module where the barcode is read. Afterwards, the vial can be transported back using TransportVialHome() or transported elsewhere using TransportVial()",
    "AspirateSyringe": "This activity aspirates a specified volume.",
    "DispenseSyringe": "This activity dispenses a specified volume.",
    "EmptySyringe": "This activity dispenses the whole content of the syringe.",
    "FillingStrokes": "This activity performs a specified number of filling strokes that can be used for bubble elimination.",
    "CleanSyringe": "Activity used to clean a syringe or the LCMS tool in a wash station.",
    "CleanInjector": "Activity used to clean the injection valve inlet port with a syringe or the LCMS tool.",
    "RinseWashLiner": "This activity rinses the liner of a Fast Wash Station or of a HF Fast Wash Station. Fast Wash Stations rinse with max flow rate. The rinse duration is calculated from the liner volume and the max flow rate. HF Fast Wash Stations rinse instead with the default flow factor and the default rinse duration from the module.",
    "InjectSampleGC": "This activity performs an injection into an injector port. All necessary parameters can be specified.",
    "FastInjectSampleGC": "This activity injects a specified volume into the injector in fast mode, which minimizes the dwell time of the needle in the injector. Penetration speed and injection flow rate are set to maximum values.",
    "InjectSampleLC": "This activity performs an injection into a valve injection port. All necessary parameters can be specified. MoveToObject needs to be executed before this activity.",
    "PenetrateWithConstForce": "The PALTool attached to the PALHead penetrates a target object with the tool, detects the bottom and contacts the bottom with a constant force (sealing of the needle).",
    "MoveSelectorValve": "Moves a selector valve from the current position to the desired destination port with a constant angular speed that is derived from the parameter switch time found selector valve type.",
    "MoveInjectorValve": "Moves the Injector Valve to the requested position and sets the output signal if defined.",
    "MoveTwoPositionNPortValve": "Moves the TwoPositionNPort Valve to the requested position and sets the output signal if defined.",
    "SetAgitator": "This activity is used to turn the agitator shaking on and off. The agitator is designed to alter the rotation direction after each pause (parameters onTime & offTime). Activity TransportVial() pauses and resumes the agitator automatically when placing or removing a vial from the agitator.",
    "SetTemperature": "This activity starts the temperature control module of the target object (e.g. Agitator, Headspace Syringe Heater, Peltier Stack).",
    "StartPurgeSyringe": "This activity purges the Headspace syringe using the internal purge gas line.",
    "StopPurgeSyringe": "This activity stops the purging of the Headspace syringe.",
    "PickToolAdapter": "The MHE ToolAdapter is mounted onto the tool with this activity.",
    "ParkToolAdapter": "The MHE ToolAdapter is parked with this activity.",
    "PurgeVial": "This activity uses the MHE adapter to purge the gas phase in a Headspace vial for a specified time. Used for active MHE.",
    "StartSpmeAdsorb": "This activity penetrates the target vial that is either placed on a Rack or in the Agitator module and exposes the SPME fibre.",
    "StopSpmeAdsorb": "The PALHead retracts the fibre and depentrates/leaves the target object.",
    "StartSpmeDesorb": "This activity is used to prepare the PALHead for bake out of the SPME fiber either in the SPME Conditioning Module or in an injector, but terminates afterwards. A temperature can be set in case of the Conditioning Module.",
    "StopSpmeDesorb": "The PALHead retracts the fibre and depentrates/leaves the target object.",
    "Beep": "This activity emits a beep for the time specified.",
    "WaitForSyncSignal": "This activity waits for the specified input signal.",
    "SetSignal": "This activity sets the specified output signal.",
    "StartTimer": "This activity starts a timer. This timer will be consumed at a later stage in the script execution.",
    "WaitForTimer": "This activity consumes a previously started timer. It waits until the specified time has been elapsed.",
    "Wait": "This activity waits for the time specified.",
    "MoveToPosition": "Moves head with tool to a given position.",
    "MoveRelative": "Moves head with tool relative to a (optionally) given reference point or relative to the actual position if no reference point is given. Attention: with forceDriectMovement=true all safety checks are disabled. This may lead to damaged hardware!",
    "MoveTorqueMode": "Moves a PAL3 axis in torque mode as a relative movement (axis bounds are checked and move requests beyond bounds are rejected). Depending on the chosen method, the axis either reaches the target or finds an impact. The effective distance travelled by the axis is then returned",
    "SetStirrer": "This activity is used to turn a stirrer on and off. The stirrer can be operated in continuous or intermittent motion though the parameters onTime & offTime. Activity TransportVial() pauses and resumes the stirrer automatically when placing into or removing an object from the stirrer.",
    "SetRotator": "This activity is used to turn an IRotator on and off and can operate any object implementing the IRotator capability (Agitators, Stirrers and VortexMixers) Most IRotator(s) can be operated in continuous or intermittent motion though the parameters onTime & offTime."
}

methods = """
    ApplyPressure( target, penetrationDepth /*, pressurizeTime, penetrationSpeed, leaveObject */ )
    AirgapTip( volume /*, flowRate */ )
    ApproachObject( target /*, index, openCoverPlate, motionFactor, motionOption, offsetX, offsetY, offsetZ */ )
    AspirateDilutor( dilutor, volume /*, flowRate, overfillRate, pullupDelay */ )
    AspirateSyringe( volume /*, flowRate, overfillRate, pullupDelay */ )
    AspirateTip( volume /*, flowRate, pullupDelay, trackLiquidLevel */ )
    AspirateTipWithLiquidClass( target, volume /*, index, useLLD */ )
    Beep( duration /*, frequency */ )
    CalibrateDeCapper( target )
    CalibrateRobotArmCurrent( /* usage */ )
    CapObject( target, capper /*, capperIndex, dispose */ )
    ChangeTool( /* tool */ )
    CleanInjector( target, washSource /*, washVolume, needleGap, dispenseFlowRate, dispensePullupDelay, washPenetrationDepth, aspirateFlowRate, aspiratePullupDelay, washIndex, leaveObjectAfterClean */ )
    CleanSyringe( washSource /*, washIndex, washPenetrationDepth, aspirateFlowRate, pullupDelay, wasteTarget, wasteIndex, wastePenetrationDepth, dispenseFlowRate, washVolume, washAirGapVolume, cycles */ )
    CloseOpenDrawers( stack )
    DecapObject( target, decapper /*, decapperIndex */ )
    DeCapperDispose( target )
    DeliverLiquidDilutor( dilutor, volume /*, aspirateFlowRate, aspiratePullupDelay, dispenseFlowRate, solventPort, deliveryPort */ )
    Depenetrate( /* wipeOff, depthReduction */ )
    DetectObject( target /*, index, coverPlateIsOpen, searchDistance, detectionCurrent */ )
    DetectObjectContainer( target /*, index, searchDistance, detectionCurrent, evaluateContainerCollision */ )
    DispenseDilutor( dilutor, volume /*, flowRate */ )
    DispenseSyringe( volume /*, flowRate */ )
    DispenseTip( volume /*, flowRate, dispenseDelay, trackLiquidLevel */ )
    DispenseTipWithLiquidClass( target, volume /*, index, useLLD */ )
    DisposeTip( /* stripForce */ )
    DisposeVial( source /*, needleTransportPenetrationDepth */ )
    EmptyDilutor( dilutor /*, flowRate */ )
    EmptySyringe( /* flowRate */ )
    EmptyTip( /* flowRate, dispenseDelay, moveToWaste */ )
    FastInjectSampleGC( injector /*, injectedSignal, waitTime, timerTolerance, waitTimer, timer, penetrationDepth, volume, accelerationDistance, penetrateOverlapDistance, depenetrateOverlapDistance, plungerRetractDelay, depenetrationDelay */ )
    FillingStrokes( /* volume, aspirateFlowRate, dispenseFlowRate, pullupDelay, dispenseDelay, count */ )
    GenericGripperGetAdapterDistance( )
    GenericGripperGrabObject( /* grabDistance, grabSpeed, retractDistanceBefore, retractDistanceAfter, absAdapterDistance, gripForce, gripDirection */ )
    GenericGripperGrip( /* gripForce, gripDirection */ )
    GenericGripperMoveAdapterAbsolute( absDistance )
    GenericGripperMoveAdapterRelative( relDistance )
    GenericGripperRelease( /* relReleaseDistance, releaseDirection */ )
    GenericGripperReleaseObject( /* retractDistanceBefore, retractDistanceAfter, relReleaseDistance, releaseDirection */ )
    GetActualVolume( target, index )
    GetAnalogSignalADCValue( signal )
    GetAnalogSignalValue( signal )
    GetArmPosition( /* part */ )
    GetAxisPosition( axis )
    GetCentrifugeRotorTemperature( target )
    GetObjectPosition( target /*, index, childTarget */ )
    GetSampleAndCleanInjector( sampleTarget, aspirateVolume, injectorTarget /*, sampleIndex, useTouchDown, leaveSample, penetrationDepth, penetrationSpeed, useBottomSense, heightFromBottom, aspirateRearAirGapVolume, aspirateFrontAirGapVolume, aspirateFlowRate, aspirateOverfillRate, aspiratePullupDelay, washPump1, washPumpIndex1, washTime1, washPump2, washPumpIndex2, washTime2 */ )
    GetSignalState( signal )
    GetStepperPosition( target )
    GrabObject( )
    GripperGrip( )
    GripperRelease( )
    GripperReset( )
    HomeStepper( target /*, direction, velocity */ )
    InjectSampleGC( injector /*, injectedSignal, penetrationDepth, penetrationSpeed, volume, flowRate, preDelay, postDelay, waitTime, timerTolerance, timer, waitTimer, injectedSignalMode */ )
    InjectSampleLC( target /*, injectedSignal1, injectedSignal2, volume, flowRate, preDelay, postDelay, timer, waitTimer, waitTime, timerTolerance */ )
    IsStepperMoving( target )
    LeaveDeCapper( target )
    LeaveObject( /* stripDistance, leaveDrawerOpen, wipeOff */ )
    LowerNeedleGuide( )
    MoveAbsolute( /* destinationX, destinationY, destinationZ, part, accelerationFactor, drfOption, forceDirectMovement */ )
    MoveInjectorValve( target, position /*, injectedSignal1, injectedSignal2 */ )
    MoveNeedleGuide( movement /*, accelerationFactor, isRelativeMovement */ )
    MovePlunger( movement /*, speed, isRelativeMovement */ )
    MoveRelative( /* referencePoint, movementX, movementY, movementZ, accelerationFactor, drfOption, forceDirectMovement */ )
    MoveSelectorValve( target, destinationPort )
    MoveStepperAbsolute( target, position /*, velocity, acceleration, waitForTerm */ )
    MoveStepperRelative( target, position /*, velocity, acceleration, waitForTerm */ )
    MoveStepperVelocity( target, velocity /*, direction, acceleration */ )
    MoveToHome( )
    MoveToObject( target /*, index, cutFoil, openCoverPlate, useTouchDown */ )
    MoveToPosition( referencePoint /*, movementX, movementY, movementZ, searchDistance, accelerationFactor, useTouchDown, forceDirectMovement */ )
    MoveTorqueMode( axis, distance /*, torqueCurrent, speed, timeout, method, steadyImpactThreshold, steadyImpactDuration, relative, posDirMoveCurrent, negDirMoveCurrent */ )
    MoveTwoPositionNPortValve( target, position /*, injectedSignal1, injectedSignal2 */ )
    MoveValveDrive( target, position /*, velocity */ )
    MultiStageInjectSampleGc( injector /*, injectedSignal, phase1Volume, phase1FlowRate, phase1PenetrationDepth, phase1PenetrationSpeed, phase2FlowRate, phase3Volume, phase3FlowRate, phase3PenetrationDepth, preDelay, postDelay, waitTime, timerTolerance, timer, waitTimer, injectedSignalMode */ )
    ObstacleAvoidanceTest( destination )
    OperateCentrifugeCover( target, position )
    OperateDeCapperBracket( target, operation )
    ParkDeCapper( target )
    ParkTool( /* slot, safe, shutdown, releaseNdlGuideAdapter */ )
    ParkToolAdapter( )
    PauseAgitator( agitator )
    PauseRotator( target )
    PauseStirrer( stirrer )
    PenetrateObject( target /*, index, depth, speed */ )
    PenetrateWithBottomSense( target /*, index, heightFromBottom, speed, searchDistance, detectionCurrent */ )
    PenetrateWithConstForce( target /*, index */ )
    PickNeedleGuideAdapter( adapterStation /*, checkAdapterMounted */ )
    PickTool( slot )
    PickToolAdapter( toolAdapter )
    PickUpTip( target /*, index, validate, pickUpForce, validationForce */ )
    PlayMelody( melodyName /*, beatFactor, playbackMode */ )
    PreFillSyringe( volume )
    PrimeDilutor( dilutor /*, wastePosition, solventPort, volume, flowRate */ )
    PulsedSpmeConditioning( conditioningTime, target /*, pulseDuration, pauseDuration, penetrationSpeed, fiberPenetrationDepth */ )
    PurgeVial( target /*, purgeTime, penetrationDepth */ )
    PushObject( target /*, index, force, searchDistance */ )
    ReadBarcode( target /*, index, home, expectedBarcode, motionFactor, motionOption, verticalClearance */ )
    ReleaseNeedleGuideAdapter( adapterStation /*, checkAdapterMounted */ )
    ReleaseObject( )
    ReleasePressure( target /*, releaseTime */ )
    Reset( )
    ResetAfterSample( )
    ResetLiquidClass( )
    ResumeAgitator( agitator )
    ResumeRotator( target )
    ResumeStirrer( stirrer )
    RetractNeedleGuide( /* distance */ )
    ReturnTip( /* dropForce, stripForce, emptyTipInWaste */ )
    RinseWashLiner( target /*, cycles */ )
    SampleRelativeTorqueModeMoveCurrent( axis, velocity )
    ScrewCap( target /*, heightCheck, heightCheckMaxDeviation */ )
    SetActualVolume( target, index, volume )
    SetAgitator( agitator /*, speed, state, onTime, offTime, timer */ )
    SetAttachmentState( target /*, state, mode, offset */ )
    SetCentrifuge( target /*, speed, gForce, state, waitForConstSpeed */ )
    SetContactState( target /*, state, offset */ )
    SetLcToolPosition( target, washSource, lcPosition )
    SetParameter( target, name /*, valueInt, valueString */ )
    SetPump( target /*, pumpIndex, flowRate, flowFactor, state */ )
    SetRotator( target /*, speed, state, onTime, offTime, waitForConstSpeed */ )
    SetScheduledSignal( signal, time )
    SetSignal( signal /*, mode */ )
    SetStandbyTemperature( target /*, standbyTemperature, commitImmediately */ )
    SetStepperPosition( target, position )
    SetStirrer( stirrer /*, speed, state, onTime, offTime */ )
    SetTemperature( target, temperature /*, tolerance, monitorTemperature, wait */ )
    SetToolState( attachable /*, attachmentState */ )
    SetValveDriveSchedule( /* target1, position1, velocity1, target2, position2, velocity2, signal1, signal2, waitTime */ )
    SetVolatilePosition( target, type, position /*, positionIndex */ )
    SetVortexMixer( vortexMixer /*, speed, state */ )
    StartBubbleDetector( target )
    StartPurgeSyringe( /* timer */ )
    StartSpmeAdsorb( target /*, index, penetrationSpeed, fiberPenetrationDepth, doAgitation */ )
    StartSpmeDesorb( target /*, penetrationSpeed, fiberPenetrationDepth */ )
    StartSpmeInject( target /*, penetrationSpeed, fiberPenetrationDepth, injectedSignal, injectedSignalMode */ )
    StartTimer( timer )
    StopBubbleDetector( target, numberOfExcpectedLevelChanges /*, bubbleFilter, compareOption */ )
    StopConstForce( )
    StopPurgeSyringe( /* volume */ )
    StopSpmeAdsorb( )
    StopSpmeDesorb( )
    StopSpmeInject( )
    StopStepper( target /*, waitForTerm */ )
    SwitchCentrifugePosition( target, index )
    SwitchDilutorValve( dilutor, valvePosition )
    SwitchGasValve( target, position )
    SwitchOffHeater( target )
    TransportVial( source, destination /*, destinationIndex, home, leaveObject, leaveDrawerOpen, needleTransportPenetrationDepth, motionFactor, motionOption */ )
    TransportVialHome( vial /*, leaveObject, leaveDrawerOpen, motionFactor, motionOption */ )
    UnscrewCap( target /*, retighten, slipCheck */ )
    UseLiquidClass( liquidClass /*, multiDispense */ )
    ValidateTemperatureControl( target )
    VortexVial( source, vortexMixer /*, vortexMixerSpeed, time, leaveDrawerOpen */ )
    Wait( time )
    WaitForSyncSignal( signal )
    WaitForTimer( timer, time /*, tolerance */ )
"""


def extract_script_contents(file_path):
    """
    Extracts and returns the script contents from a given file path.

    Args:
    - file_path: A string representing the path to the file containing the scripts.

    Returns:
    - A list of strings, where each string is the content of a script found in the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        # Safely evaluate the string as a Python literal
        scripts = ast.literal_eval(content)
    except Exception as e:
        print(f"Error reading or evaluating the file: {e}")
        return []

    script_contents = []
    for script in scripts:
        if 'metadata' in script and 'script' in script['metadata']:
            script_contents.append(script['metadata']['script'])

    return script_contents


# Function to generate embeddings
def embed_text(text, model="text-embedding-3-large"):
   return client.embeddings.create(input=[text], model=model).data[0].embedding


# Function to query Pinecone for relevant documents
def query_pinecone(query_text, top_k=7):
    query_embedding = embed_text(query_text)
    # Use keyword arguments for the query
    query_results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return query_results["matches"]


# Simple response generation based on document IDs
def generate_response(matches):
    if not matches:
        return "I couldn't find any relevant information in the documents."

    doc_ids = [match["id"] for match in matches]
    response = "Based on the documents I found, you might want to look at: " + ", ".join(doc_ids)
    return response


# The main chat function
def chat_with_user(query):
    matches = query_pinecone(query, top_k_select)
    if matches:
        response = generate_response_with_openai(query, matches)
    else:
        response = "I'm sorry, I couldn't find any relevant information."
    return response


def query_and_respond(query):
    # Simple response to confirm function is called
    return f"Received query: {query}"


def generate_xml(chat_script):
    # ... Logic to convert your chat_script to a valid XML structure ...
    xml_data = """
    XML Goes Here
    """
    return xml_data


def generate_response_with_openai(query, matches):

    with open('vector_results.txt', 'w+', encoding='utf-8') as f:
        f.write(str(matches))

    # Constructing context for the GPT model. This could be improved by fetching document contents or summaries.
    documents_context = "Based on the following information/scripts " + "; \n\n".join(extract_script_contents('vector_results.txt'))
    prompting = f"""You are an expert in PAL3 scripting for laboratory analysis. {documents_context}
    \n\nUse these scripts to generate a new script to perform the following task'{query}'.
    
    Extract all the relevant parameters, constants variables and method sequences and use only the required parameters for the script.
    You will likely need to extract certain bits of each script and combine them to form a new script.
        
    All responses must be in the format
    
    <example>
    procedure Main(
    insert all required parameters here e.g. 
    liquidSourceRack:IRack,
    liquidSourceIndex:Integer[1..100]=1,
    syringe:ITool,
    transferVolume:Volume[1uL..10000uL]=1uL,
    
    )
    
    const
    insert all constants here
    
    var
    insert all variables here 
    
    begin
    Put all steps here e.g.
    MoveToObject(liquidSourceRack, liquidSourceIndex)
    AspirateSyringe(transferVolume)
    end
    
    add additional methods below if necessary e.g.
    
    procedure USPE(
    insert all required parameters here e.g. 
    liquidSourceRack:IRack,
    liquidSourceIndex:Integer[1..100]=1,
    syringe:ITool,
    transferVolume:Volume[1uL..10000uL]=1uL,
    ...........
    ..........
    
    
    const
    insert all constants here
    
    var
    insert all variables here 
    
    begin
    Put all steps here e.g.
    MoveToObject(liquidSourceRack, liquidSourceIndex)
    AspirateSyringe(transferVolume)
    end
    )
    </example>
    
    Check these definitions for any terms you may not understand here:<definitions>\n\n{glossary} \n {methods}\n\n</definitions>
    
    You must return the script in the format above with no explanations or additional information. It is critical that the script is correct and complete to ensure the instrument doesnt break.
    """
    print(prompting)
    # get the number of tokens in the prompt using tiktoken
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
    st.write(f"Number of tokens in the prompt: {len(encoding.encode(prompting))}")

    st.session_state.messages.append({"role": "user", "content": query})

    messages = [{"role": "system", "content": prompting}] + st.session_state.messages

    response = client.chat.completions.create(
        model='gpt-4o-2024-08-06',
        messages=messages,
        stream=False,
        temperature=TEMPERATURE,
        max_tokens=3000
    )

    st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})

    return response.choices[0].message.content

# Set up the page
st.set_page_config(
    page_title="PAL Chat Script Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set up the title and description of the app
st.title("PAL Chat Script Generator")

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False

    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


if check_password():
    st.write(f'Welcome *{st.secrets["username"]}*')


    st.write(
        """This app generates a PAL script using LangChain, OpenAI and Pinecone. 
        
        Type you request below detailing what you want the PAL rail to do and click the button to generate a PAL script.
        
        To get the results please use the following format:
        
        Summary: Write a brief summary of what you want to do, 1 to 2 sentences. Then list the steps in a bit more detail.
        
        1. Get the 2500ul tool
        2. Add 2000ul of solvent X into each vial
        3. .....
        4. .....
        """
    )

    # Set up the sidebar
    st.sidebar.title("Settings")
    if st.sidebar.button('Log Out'):
        st.session_state["password_correct"] = False
        st.session_state.clear()
        st.rerun()
    st.sidebar.write("Select the settings for the chat script generation.")
    # set up a selection box for the top_k results
    top_k_select = st.sidebar.number_input(
        "Top K Results",
        min_value=5,
        max_value=10,
        value=5,
        help="""The number of top documents to retrieve from Pinecone. This is performed by performing a sematic search.
             The higher the number, the more accurate the response will be, but it will take longer to generate the response.
             The default value is 30. Please note that high values may breach the upper context limit for the LLM."""
    )

    # Set up the chat script generation
    st.header("Chat Script Generation")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = ''

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # streamlit chat_message
    prompt = st.chat_input("Enter your chat input here:")
    if prompt:
        with st.spinner("Generating chat script..."):
            with st.chat_message("user"):
                st.write(prompt)
            chat_script = chat_with_user(prompt)
            st.code(chat_script, language='pall')
            st.info("The chat script has been generated successfully. Please feel free to ask additional questions in the input above\
                    The previous chat script will be stored in the memory and used to generate the next chat script.")

            # Download Button
            st.download_button(
                label="Download as XML",
                data=generate_xml(chat_script),  # Function to generate XML data
                file_name="chat_script.xml",
                mime="text/xml"
            )

    # Set up the footer
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app generates a PAL script for Thermo autosampler rail based on natural language input\
        using LangChain, OpenAI, Pinecone, and HuggingFace.
        """
    )

    # Set up the footer
    st.sidebar.title("Help")
    st.sidebar.info(
        """
        For help or support, please contact
        Daniel Halwell.
        """
    )

    # Set up the footer
    st.sidebar.title("Feedback")
    st.sidebar.info(
        """
        If you have any feedback or suggestions for this app, please contact
        Daniel Halwell.
        """
    )

    # Set up the footer
    st.sidebar.title("Connect")
    st.sidebar.info(
        """
        Connect with Daniel Halwell and review the code on GitHub for more information:
        - [GitHub](https://github.com/azu-business/chat_pall_scr)
        """
    )