import React, {Component} from 'react';
import PropTypes from 'prop-types';


class PlaneSVG extends Component {

  constructor(props) {
    super(props);
    this.boundingBoxSpecified = this.boundingBoxSpecified.bind(this);
  }

  createLineElement() {
    return (
        <line x1="0"
              y1={this.props.lineIndex}
              x2={this.props.width}
              y2={this.props.lineIndex}
              style={{
                stroke: 'white',
                strokeWidth: 2,
              }}
        />
    );
  }

  boundingBoxSpecified() {
    return (this.props.roiX1 && this.props.roiY1 &&
            this.props.roiX2 && this.props.roiY2)
  }

  render() {
    return (
        <svg height={this.props.height}
             width={this.props.width}
             onWheel={this.props.scrollEvent}
             onMouseDown={this.props.mouseDownEvent}
             onMouseMove={this.props.mouseMoveEvent}
             onMouseUp={this.props.mouseUpEvent}
             className='svg'
        >
          <image
              href={`/image/${this.props.viewType}/${this.props.patientId}/${this.props.posIndex}`}/>
          {this.props.lineIndex ? this.createLineElement() : null}
          {this.boundingBoxSpecified() &&
            <line x1={this.props.roiX1}
              y1={this.props.roiY1}
              x2={this.props.roiX2}
              y2={this.props.roiY1}
              style={{
                stroke: this.props.colorX,
                strokeWidth: 2,
              }}
            />
          }
          {this.boundingBoxSpecified() &&
            <line x1={this.props.roiX1}
                y1={this.props.roiY2}
                x2={this.props.roiX2}
                y2={this.props.roiY2}
                style={{
                  stroke: this.props.colorX,
                  strokeWidth: 2,
                }}
            />
          }
          {this.boundingBoxSpecified() &&
            <line x1={this.props.roiX1}
                y1={this.props.roiY1}
                x2={this.props.roiX1}
                y2={this.props.roiY2}
                style={{
                  stroke: this.props.colorY,
                  strokeWidth: 2,
                }}
            />
          }
          {this.boundingBoxSpecified() &&
            <line x1={this.props.roiX2}
                y1={this.props.roiY1}
                x2={this.props.roiX2}
                y2={this.props.roiY2}
                style={{
                  stroke: this.props.colorY,
                  strokeWidth: 2,
                }}
            />
          }
        </svg>
    )
  }
}

PlaneSVG.propTypes = {
  viewType: PropTypes.string.isRequired,
  patientId: PropTypes.string.isRequired,
  width: PropTypes.number.isRequired,
  height: PropTypes.number.isRequired,
  colorX: PropTypes.any.isRequired,
  colorY: PropTypes.any.isRequired,
  roiX1: PropTypes.number.isRequired,
  roiX2: PropTypes.number.isRequired,
  roiY1: PropTypes.number.isRequired,
  roiY2: PropTypes.number.isRequired,
  posIndex: PropTypes.number.isRequired,
  lineIndex: PropTypes.number,
  scrollEvent: PropTypes.func.isRequired,
};


export default PlaneSVG;
